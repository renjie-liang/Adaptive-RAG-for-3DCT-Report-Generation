"""
Dataset for ViSD-Boost embeddings and VQA tasks

ViSDVQADataset: NPZ + VQA JSON (report_generation, long_answer, short_answer, multiple_choice).
Single dataloader; prompt/target per sample; task_type for eval split.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from adaragct.utils.logger import get_module_logger


def trunc_str(s, n=200):
    """Truncate string for debug printing. Single definition for use in evaluate/train_step."""
    s = str(s)
    return s[:n] + "..." if len(s) > n else s


# Fixed semantic slot names (used in prompts; independent of NPZ key names)
CUE_SLOT_NAMES = ["whole_ct", "lung", "heart", "esophagus", "aorta"]
NUM_EMBEDDINGS = len(CUE_SLOT_NAMES)

# Import organ tokens from constants
from adaragct.constants import ORGAN_TOKENS, PROVIDED_TOKEN

# New prompt format: organ tokens on separate lines (each will be replaced by embedding)
# This replaces the old CUE_TEXT which used multiple <image> tokens
# ORGAN_TOKENS_STR = "\n".join(ORGAN_TOKENS)  # "<whole_ct>\n<lung>\n<heart>\n<esophagus>\n<aorta>"
# ORGAN_TOKENS_DESC = (
#     "Below are 5 visual embeddings extracted from the same chest CT volume:\n"
#     "- Whole CT: <whole_ct>\n"
#     "- Lungs: <lung>\n"
#     "- Heart: <heart>\n"
#     "- Esophagus: <esophagus>\n"
#     "- Aorta: <aorta>"
# )

ORGAN_TOKENS_DESC = (
    "I will provide 5 embeddings from the same CT volume, representing different anatomical regions: "
    "whole CT <whole_ct>, lungs <lung>, heart <heart>, esophagus <esophagus>, and aorta <aorta>."
)

# Provided token description for paper samples (placeholder only)
PROVIDED_TOKEN_DESC = (
    "This is a text-only medical case with no visual embeddings provided <provided>."
)

# CT-RATE 18 abnormality label names (column order in multi_abnormality_labels CSV)
CTRATE_LABEL_NAMES = [
    'Medical material', 'Arterial wall calcification', 'Cardiomegaly',
    'Pericardial effusion', 'Coronary artery wall calcification', 'Hiatal hernia',
    'Lymphadenopathy', 'Emphysema', 'Atelectasis', 'Lung nodule', 'Lung opacity',
    'Pulmonary fibrotic sequela', 'Pleural effusion', 'Mosaic attenuation pattern',
    'Peribronchial thickening', 'Consolidation', 'Bronchiectasis',
    'Interlobular septal thickening',
]


def _format_labels_as_text(label_row: dict) -> str:
    """Convert a label dict {name: 0/1} to a clinical findings text string."""
    positives = [name for name in CTRATE_LABEL_NAMES if label_row.get(name, 0) == 1]
    negatives = [name for name in CTRATE_LABEL_NAMES if label_row.get(name, 0) == 0]
    pos_str = ', '.join(positives) if positives else 'none'
    neg_str = ', '.join(negatives) if negatives else 'none'
    return f"Detected findings: {pos_str}. Not detected: {neg_str}."


# System prompt presets for chat template
SYSTEM_PROMPTS = {
    "minimal": "You are a radiologist.",
    "detailed": (
        "You are an experienced radiologist specializing in chest CT interpretation. "
        "Generate a detailed findings report covering all anatomical structures."
    ),
    "structured": (
        "You are a radiologist. Report findings for each anatomical region: "
        "lungs, heart, esophagus, aorta, and other structures."
    ),
}


def _build_chat_prompt(system_prompt: str, user_content: str, assistant_content: str = None) -> tuple:
    """Build Llama 3 chat template prompt.
    
    Returns:
        If assistant_content is None: (prompt_text,) — just the prompt part
        If assistant_content is given: (prompt_text, target_text) — prompt and target
        
    The prompt_text includes everything up to and including the assistant header.
    The target_text is the assistant response + <|eot_id|>.
    """
    prompt_text = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if assistant_content is not None:
        target_text = f"{assistant_content}<|eot_id|>"
        return prompt_text, target_text
    return (prompt_text,)


# ========== Organ Report Mode (B/C/D) constants ==========
ORGAN_ORDER = ["lung", "heart", "esophagus", "aorta", "other"]

DEFAULT_NEGATIVES = {
    "lung": "The lungs are clear without suspicious nodules, consolidation, or pleural effusion. Trachea and main bronchi are patent.",
    "heart": "Heart size and mediastinal vascular structures are within normal limits. No pericardial effusion is seen.",
    "esophagus": "The thoracic esophagus shows normal calibration without wall thickening.",
    "aorta": "The thoracic aorta has a normal diameter without aneurysmal dilatation or significant calcification.",
    "other": "No significant abnormalities are seen in the visualized abdominal organs and bone structures.",
}

ORGAN_SYSTEM_PROMPT = (
    "You are a radiology report generator for chest CT scans. "
    "Given anatomical embeddings from a CT volume, generate accurate radiology findings "
    "for the specified organ(s). Output ONLY the clinical findings text, nothing else."
)


def _build_organ_samples(
    raw_data: list,
    image_to_idx: dict,
    embeddings_per_key: dict,
    embedding_keys: list,
    mode: str,
    single_organ_ratio: float,
    use_chat_template: bool,
    system_prompt: str,
) -> list:
    """Build samples list from organ_reports JSON for modes B (single), C (full), D (mixed).

    Each returned sample is a tuple:
        (emb_tensor, prompt, target, task_type, sample_id, source_id, has_image, question, raw_target)

    raw_target: original organ text (empty string if organ had no annotation) — used as reference_text in eval.
    target: DEFAULT_NEGATIVES-filled version — used for training loss (never empty).
    """
    import random as _random
    samples = []

    for item in raw_data:
        image_id = str(item["image"])
        norm_id = _normalize_image_id(image_id)
        if norm_id not in image_to_idx:
            continue
        npz_idx = image_to_idx[norm_id]
        emb_list = [embeddings_per_key[k][npz_idx] for k in embedding_keys]
        emb_tensor = torch.stack(emb_list, dim=0)

        organ_reports = item.get("organ_reports", {})

        if mode == "single":
            _add_single_organ_samples(
                samples, item, emb_tensor, image_id, organ_reports,
                use_chat_template, system_prompt
            )
        elif mode == "full":
            _add_full_block_sample(
                samples, item, emb_tensor, image_id, organ_reports,
                use_chat_template, system_prompt
            )
        elif mode == "mixed":
            if _random.random() < single_organ_ratio:
                _add_single_organ_samples(
                    samples, item, emb_tensor, image_id, organ_reports,
                    use_chat_template, system_prompt
                )
            else:
                _add_full_block_sample(
                    samples, item, emb_tensor, image_id, organ_reports,
                    use_chat_template, system_prompt
                )

    return samples


def _add_single_organ_samples(samples, item, emb_tensor, image_id, organ_reports,
                               use_chat_template, system_prompt):
    """Mode B: one sample per organ, with GT context from preceding organs."""
    for organ_idx, target_organ in enumerate(ORGAN_ORDER):
        context_parts = []
        for prev_organ in ORGAN_ORDER[:organ_idx]:
            prev_text = organ_reports.get(prev_organ, "").strip()
            filled = prev_text or DEFAULT_NEGATIVES[prev_organ]
            context_parts.append(f"[{prev_organ.upper()}] {filled}")
        context_str = "\n".join(context_parts)

        raw_target = organ_reports.get(target_organ, "").strip()
        train_target = raw_target or DEFAULT_NEGATIVES[target_organ]

        if context_str:
            user_body = (
                f"{ORGAN_TOKENS_DESC}\n"
                f"Previous findings:\n{context_str}\n\n"
                f"Generate findings ONLY for the {target_organ.upper()}."
            )
        else:
            user_body = (
                f"{ORGAN_TOKENS_DESC}\n"
                f"Generate findings ONLY for the {target_organ.upper()}."
            )

        if use_chat_template:
            prompt, target_text = _build_chat_prompt(system_prompt, user_body, train_target)
            raw_target_text = raw_target  # keep raw for eval
        else:
            prompt = user_body
            target_text = train_target
            raw_target_text = raw_target

        sample_id = f"{item['id']}_{target_organ}"
        samples.append((
            emb_tensor, prompt, target_text, "report_generation",
            sample_id, image_id, True, f"Generate findings for {target_organ}", raw_target_text
        ))


def _add_full_block_sample(samples, item, emb_tensor, image_id, organ_reports,
                            use_chat_template, system_prompt):
    """Mode C: one sample = full organ block [LUNG]...\n[HEART]..."""
    train_parts = []
    raw_parts = []
    for organ in ORGAN_ORDER:
        raw_text = organ_reports.get(organ, "").strip()
        filled = raw_text or DEFAULT_NEGATIVES[organ]
        train_parts.append(f"[{organ.upper()}] {filled}")
        raw_parts.append(f"[{organ.upper()}] {raw_text}" if raw_text else f"[{organ.upper()}]")

    train_target = "\n".join(train_parts)
    raw_target = "\n".join(raw_parts)

    user_body = (
        f"{ORGAN_TOKENS_DESC}\n"
        "Generate a complete radiology report covering all anatomical regions, "
        "starting from LUNG."
    )

    if use_chat_template:
        prompt, target_text = _build_chat_prompt(system_prompt, user_body, train_target)
        raw_target_text = raw_target
    else:
        prompt = user_body
        target_text = train_target
        raw_target_text = raw_target

    sample_id = f"{item['id']}_full"
    samples.append((
        emb_tensor, prompt, target_text, "report_generation",
        sample_id, image_id, True, "Generate complete radiology report", raw_target_text
    ))


# Task token suffixes (after cue) for VQA
TASK_TOKENS = {
    "report_generation": "<report_generation>",
    "long_answer": "<long_answer>",
    "short_answer": "<short_answer>",
    "multiple_choice": "<multiple_choice>",
}

def _normalize_image_id(image_id: str) -> str:
    """Normalize image id for lookup (strip .nii.gz)."""
    s = str(image_id).strip()
    if s.endswith(".nii.gz"):
        return s[:-7]
    return s


def _task_from_id(item_id: str) -> str:
    """Derive task type from VQA item id (e.g. report_generation_0 -> report_generation)."""
    parts = item_id.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]
    return item_id


def _task_from_conversation_type(conv_type: str) -> str:
    """Map conversation type to task (CT-CHAT style)."""
    if conv_type in ("free_response", "description", "conversation"):
        return "long_answer"
    if conv_type == "multiple_choice":
        return "multiple_choice"
    if conv_type == "report_generation":
        return "report_generation"
    return "short_answer"  # short_answer, conversation_questions, only_text, etc.


def _strip_question_text(human_value: str, task_type: str) -> str:
    """Remove <image> and trailing task tokens from human value to get question text."""
    text = human_value.replace("<image>\n", "").replace("<image>", "").strip()
    for tok in TASK_TOKENS.values():
        text = text.replace(tok, "").strip()
    return text


class ViSDVQADataset(Dataset):
    """
    Dataset from VQA JSON + NPZ: one list of samples (report_generation, long_answer, short_answer, multiple_choice).
    Each sample: (embeddings, prompt_text, target_text, task_type). Prompt does not include image tokens (added in train/eval).
    """

    def __init__(
        self,
        embeddings_npz: str,
        vqa_json_path: Optional[str] = None,
        train_tasks: Optional[List[str]] = None,
        embedding_keys: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        sample_types: Optional[List[str]] = None,  # ["image"], ["paper"], ["image", "paper"]
        use_chat_template: bool = False,
        system_prompt: str = "minimal",
        whole_ct_npz: Optional[str] = None,
        whole_ct_dim: Optional[int] = None,
        oversample_rare_findings: bool = False,
        oversample_min_positives: int = 3,
        oversample_factor: float = 3.0,
        label_in_prompt: bool = False,
        label_csv_path: Optional[str] = None,
        organ_report_json: Optional[str] = None,
        organ_report_mode: Optional[str] = None,  # None | "single" | "full" | "mixed"
        single_organ_ratio: float = 0.4,
    ):
        logger = get_module_logger()
        logger.info("========== ViSDVQADataset.__init__ START ==========")
        logger.info(f"embeddings_npz: {embeddings_npz}")
        logger.info(f"vqa_json_path: {vqa_json_path}")
        logger.info(f"train_tasks: {train_tasks}")
        logger.info(f"max_samples: {max_samples}")
        logger.info(f"sample_types: {sample_types}")
        logger.info(f"use_chat_template: {use_chat_template}")
        logger.info(f"system_prompt: {system_prompt}")
        logger.info(f"whole_ct_npz: {whole_ct_npz}")
        logger.info(f"whole_ct_dim: {whole_ct_dim}")
        logger.info(f"oversample_rare_findings: {oversample_rare_findings} (min_positives={oversample_min_positives}, factor={oversample_factor})")
        logger.info(f"label_in_prompt: {label_in_prompt} (csv={label_csv_path})")
        logger.info(f"organ_report_mode: {organ_report_mode} (json={organ_report_json}, single_organ_ratio={single_organ_ratio})")

        self.oversample_rare_findings = oversample_rare_findings
        self.oversample_min_positives = oversample_min_positives
        self.oversample_factor = oversample_factor
        self.label_in_prompt = label_in_prompt
        self.use_chat_template = use_chat_template
        # Resolve system prompt: use preset if key matches, otherwise treat as custom string
        if use_chat_template:
            self.system_prompt = SYSTEM_PROMPTS.get(system_prompt, system_prompt)
            logger.info(f"Resolved system_prompt: {self.system_prompt}")
        else:
            self.system_prompt = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

        # ========== Organ Report Mode (B/C/D): bypass VQA JSON entirely ==========
        if organ_report_mode is not None:
            if organ_report_json is None:
                raise ValueError("organ_report_json is required when organ_report_mode is set")
            if organ_report_mode not in ("single", "full", "mixed"):
                raise ValueError(f"organ_report_mode must be 'single', 'full', or 'mixed', got: {organ_report_mode}")
            if embedding_keys is None:
                raise ValueError("embedding_keys is required")

            self.embedding_keys = list(embedding_keys)
            self.sample_types = sample_types or ["image"]
            path_npz = Path(embeddings_npz)
            if not path_npz.exists():
                raise FileNotFoundError(f"Embeddings file not found: {path_npz}")

            data = np.load(embeddings_npz, allow_pickle=True)
            sample_ids = data["sample_ids"]
            if hasattr(sample_ids, "tolist"):
                sample_ids = sample_ids.tolist()
            n_raw = len(sample_ids)
            logger.info(f"NPZ sample_ids count: {n_raw}")

            whole_ct_key = self.embedding_keys[0]
            organ_keys_only = self.embedding_keys[1:]

            if whole_ct_npz is not None:
                wct_data = np.load(whole_ct_npz, allow_pickle=True)
                wct_sample_ids = wct_data["sample_ids"]
                if hasattr(wct_sample_ids, "tolist"):
                    wct_sample_ids = wct_sample_ids.tolist()
                wct_id_to_idx = {_normalize_image_id(str(sid)): i for i, sid in enumerate(wct_sample_ids)}
                wct_emb_raw = torch.from_numpy(np.asarray(wct_data[whole_ct_key])).float()
                organ_emb_raw = {k: torch.from_numpy(np.asarray(data[k])).float() for k in organ_keys_only}
                organ_dim = organ_emb_raw[organ_keys_only[0]].shape[1]
                wct_dim = wct_emb_raw.shape[1]
                max_emb_dim = max(organ_dim, wct_dim)
                self.embedding_dim_per_slot = [wct_dim] + [organ_dim] * len(organ_keys_only)
                embeddings_per_key = {}
                for k in organ_keys_only:
                    emb = organ_emb_raw[k]
                    if emb.shape[1] < max_emb_dim:
                        emb = torch.nn.functional.pad(emb, (0, max_emb_dim - emb.shape[1]))
                    embeddings_per_key[k] = emb
                wct_aligned = torch.zeros(n_raw, max_emb_dim)
                for i in range(n_raw):
                    sid = _normalize_image_id(str(sample_ids[i]))
                    if sid in wct_id_to_idx:
                        wct_aligned[i, :wct_dim] = wct_emb_raw[wct_id_to_idx[sid]]
                embeddings_per_key[whole_ct_key] = wct_aligned
            else:
                embeddings_per_key = {k: torch.from_numpy(np.asarray(data[k])).float() for k in self.embedding_keys}
                first_dim = embeddings_per_key[self.embedding_keys[0]].shape[1]
                self.embedding_dim_per_slot = [first_dim] * len(self.embedding_keys)

            image_to_idx: dict = {}
            for i in range(n_raw):
                sid = sample_ids[i] if isinstance(sample_ids[i], str) else str(sample_ids[i])
                image_to_idx[_normalize_image_id(sid)] = i

            with open(organ_report_json, "r") as f:
                organ_raw_data = json.load(f)
            logger.info(f"Organ report JSON loaded: {len(organ_raw_data)} items")

            resolved_system = ORGAN_SYSTEM_PROMPT if not use_chat_template else self.system_prompt
            self.samples = _build_organ_samples(
                raw_data=organ_raw_data,
                image_to_idx=image_to_idx,
                embeddings_per_key=embeddings_per_key,
                embedding_keys=self.embedding_keys,
                mode=organ_report_mode,
                single_organ_ratio=single_organ_ratio,
                use_chat_template=use_chat_template,
                system_prompt=self.system_prompt,
            )
            self.sample_weights = None
            logger.info(f"Organ mode '{organ_report_mode}': built {len(self.samples)} samples")
            logger.info("========== ViSDVQADataset.__init__ END (organ mode) ==========")
            return

        # ========== Standard VQA mode (existing logic below) ==========
        # Validate sample_types parameter
        if sample_types is None:
            raise ValueError("sample_types parameter is required. Must be one of: ['image'], ['paper'], ['image', 'paper']")
        valid_sample_types = ["image", "paper"]
        for st in sample_types:
            if st not in valid_sample_types:
                raise ValueError(f"Invalid sample_type '{st}'. Must be one of: {valid_sample_types}")
        if train_tasks is None:
            raise ValueError("train_tasks is required in standard VQA mode")
        if embedding_keys is None:
            raise ValueError("embedding_keys is required")
        if vqa_json_path is None:
            raise ValueError("vqa_json_path is required in standard VQA mode (organ_report_mode is None)")

        allowed: Set[str] = set(train_tasks)
        self.embedding_keys = list(embedding_keys)
        self.sample_types = sample_types
        path_npz = Path(embeddings_npz)
        path_vqa = Path(vqa_json_path)
        if not path_npz.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path_npz}")
        if not path_vqa.exists():
            raise FileNotFoundError(f"VQA JSON not found: {path_vqa}")

        allowed: Set[str] = set(train_tasks)
        self.embedding_keys = list(embedding_keys)  # NPZ keys for loading (e.g. whole_ct_attention vs whole_ct_weighted)
        logger.info(f"Allowed tasks set: {allowed}")
        logger.info(f"embedding_keys (NPZ): {self.embedding_keys}")
        
        data = np.load(embeddings_npz, allow_pickle=True)
        logger.info(f"NPZ keys: {list(data.keys())}")
        
        sample_ids = data["sample_ids"]
        if hasattr(sample_ids, "tolist"):
            sample_ids = sample_ids.tolist()
        n_raw = len(sample_ids)
        logger.info(f"NPZ sample_ids count: {n_raw}")
        logger.info(f"NPZ sample_ids first 5: {sample_ids[:5]}")
        
        # Determine which keys come from the main npz vs separate whole_ct npz
        whole_ct_key = self.embedding_keys[0]  # first key is always whole_ct
        organ_keys_only = self.embedding_keys[1:]  # remaining are organ keys
        
        if whole_ct_npz is not None:
            # Load whole_ct from separate npz (e.g. CT-CLIP)
            wct_data = np.load(whole_ct_npz, allow_pickle=True)
            logger.info(f"Separate whole_ct NPZ keys: {list(wct_data.keys())}")
            wct_sample_ids = wct_data["sample_ids"]
            if hasattr(wct_sample_ids, "tolist"):
                wct_sample_ids = wct_sample_ids.tolist()
            logger.info(f"whole_ct NPZ sample_ids count: {len(wct_sample_ids)}")
            
            # Build whole_ct id -> index mapping
            wct_id_to_idx = {}
            for i, sid in enumerate(wct_sample_ids):
                wct_id_to_idx[_normalize_image_id(str(sid))] = i
            
            wct_emb_raw = torch.from_numpy(np.asarray(wct_data[whole_ct_key])).float()
            logger.info(f"whole_ct from separate NPZ: key={whole_ct_key}, shape={wct_emb_raw.shape}")
            
            # Load organ embeddings from main npz
            organ_emb_raw = {k: torch.from_numpy(np.asarray(data[k])).float() for k in organ_keys_only}
            organ_dim = organ_emb_raw[organ_keys_only[0]].shape[1]
            wct_dim = wct_emb_raw.shape[1]
            max_emb_dim = max(organ_dim, wct_dim)
            self.embedding_dim_per_slot = [wct_dim] + [organ_dim] * len(organ_keys_only)
            logger.info(f"Heterogeneous dims: whole_ct={wct_dim}, organs={organ_dim}, max={max_emb_dim}")
            
            # Align by sample_id: use main npz sample_ids as base, lookup whole_ct by id
            embeddings_per_key = {}
            for k in organ_keys_only:
                emb = organ_emb_raw[k]
                if emb.shape[1] < max_emb_dim:
                    emb = torch.nn.functional.pad(emb, (0, max_emb_dim - emb.shape[1]))
                embeddings_per_key[k] = emb
            
            # Build whole_ct aligned to main npz sample_ids (with zero-padding if dim differs)
            wct_aligned = torch.zeros(n_raw, max_emb_dim)
            wct_found = 0
            for i in range(n_raw):
                sid = _normalize_image_id(str(sample_ids[i]))
                if sid in wct_id_to_idx:
                    wct_aligned[i, :wct_dim] = wct_emb_raw[wct_id_to_idx[sid]]
                    wct_found += 1
            embeddings_per_key[whole_ct_key] = wct_aligned
            logger.info(f"whole_ct aligned: {wct_found}/{n_raw} samples matched")
            if wct_found < n_raw:
                logger.info(f"WARNING: {n_raw - wct_found} samples missing whole_ct embedding (will be zeros)")
        else:
            # Standard: all embeddings from same npz
            embeddings_per_key = {k: torch.from_numpy(np.asarray(data[k])).float() for k in self.embedding_keys}
            first_dim = embeddings_per_key[self.embedding_keys[0]].shape[1]
            self.embedding_dim_per_slot = [first_dim] * len(self.embedding_keys)
        
        logger.info("Embedding shapes and dtypes:")
        for k in self.embedding_keys:
            emb = embeddings_per_key[k]
            logger.info(f"  {k}: shape={emb.shape}, dtype={emb.dtype}, min={emb.min():.4f}, max={emb.max():.4f}")
            if emb.shape[0] != n_raw:
                raise ValueError(
                    f"NPZ length mismatch: sample_ids has {n_raw} rows but {k} has {emb.shape[0]}"
                )

        # image_id (normalized) -> npz row index
        image_to_idx: dict = {}
        for i in range(n_raw):
            sid = sample_ids[i] if isinstance(sample_ids[i], str) else str(sample_ids[i])
            image_to_idx[_normalize_image_id(sid)] = i
        
        logger.info(f"image_to_idx mapping size: {len(image_to_idx)}")
        logger.info(f"image_to_idx first 5: {dict(list(image_to_idx.items())[:5])}")

        with open(vqa_json_path, "r") as f:
            vqa_list = json.load(f)
        logger.info(f"VQA JSON loaded, total items: {len(vqa_list)}")
        
        if not isinstance(vqa_list, list):
            raise TypeError(f"VQA JSON root must be a list, got {type(vqa_list).__name__}")
        if len(vqa_list) > 0:
            first = vqa_list[0]
            logger.info(f"VQA first item keys: {list(first.keys())}")
            logger.info(f"VQA first item id: {first.get('id')}")
            logger.info(f"VQA first item image: {first.get('image')}")
            logger.info(f"VQA first item conversations count: {len(first.get('conversations', []))}")
            for key in ("id", "image", "conversations"):
                if key not in first:
                    raise KeyError(f"VQA item missing required key '{key}'; keys: {list(first.keys())}")
            if not isinstance(first["conversations"], list):
                raise TypeError(f"VQA item 'conversations' must be a list, got {type(first['conversations']).__name__}")
        
        # Count tasks in VQA before filtering
        task_counter_before = Counter()
        for item in vqa_list:
            item_id = item.get("id", "")
            task_counter_before[_task_from_id(item_id)] += 1
        logger.info(f"VQA task breakdown (before filtering): {dict(task_counter_before)}")

        self.samples: List[Tuple[torch.Tensor, str, str, str, str, str, bool]] = []
        missing_images = []
        for item in vqa_list:
            item_id = item.get("id", "")
            task_from_id = _task_from_id(item_id)
            if task_from_id not in allowed:
                continue
            
            # Handle both image and paper samples based on sample_types parameter
            if "image" in item and "image" in self.sample_types:
                # Image sample: use real embeddings
                image = item["image"]
                img_norm = _normalize_image_id(image)
                if img_norm not in image_to_idx:
                    missing_images.append(image)
                    continue
                npz_idx = image_to_idx[img_norm]
                emb_list = [embeddings_per_key[k][npz_idx] for k in self.embedding_keys]
                emb_tensor = torch.stack(emb_list, dim=0)
                has_image = True
                source_id = image
            elif "paper" in item and "paper" in self.sample_types:
                # Paper sample: use zero embeddings (like stage3_conversation)
                max_dim = max(self.embedding_dim_per_slot)
                emb_tensor = torch.zeros(len(self.embedding_keys), max_dim)
                has_image = False
                source_id = item["paper"]
            else:
                # Skip items that don't match the specified sample_types
                raise ValueError(f"Item {item_id} has neither 'image' nor 'paper' key, but sample_types is {self.sample_types}")
                continue
            
            conversations = item["conversations"]
            for k in range(0, len(conversations) - 1, 2):
                if conversations[k]["from"] != "human" or conversations[k + 1]["from"] != "gpt":
                    continue
                human = conversations[k]
                gpt = conversations[k + 1]
                task_type = _task_from_conversation_type(human["type"])
                if task_type not in allowed:
                    continue
                question = _strip_question_text(human["value"], task_type)
                token = TASK_TOKENS[task_type]
                
                # Adjust prompt based on whether we have image
                if self.use_chat_template:
                    # Chat template mode: wrap in Llama 3 format
                    if has_image:
                        user_content = f"{ORGAN_TOKENS_DESC}\n{token}\n{question}"
                    else:
                        user_content = f"{PROVIDED_TOKEN_DESC}\n{token}\n{question}"
                    prompt_text, target_text = _build_chat_prompt(
                        self.system_prompt, user_content, gpt["value"]
                    )
                else:
                    # Legacy raw text mode (no chat template)
                    if has_image:
                        prompt_text = f"{ORGAN_TOKENS_DESC}\n{token}\n{question}"
                    else:
                        prompt_text = f"{PROVIDED_TOKEN_DESC}\n{token}\n{question}"
                    target_text = gpt["value"]
                sample_id = f"{item_id}_turn{k//2}"
                self.samples.append((emb_tensor, prompt_text, target_text, task_type, sample_id, source_id, has_image, question))

        if missing_images:
            logger.info(f"WARNING: {len(missing_images)} images not found in NPZ")
            logger.info(f"Missing images (first 10): {missing_images[:10]}")

        logger.info(f"Total samples built (before max_samples): {len(self.samples)}")
        
        task_counter_after = Counter(s[3] for s in self.samples)
        logger.info(f"Task breakdown (after filtering): {dict(task_counter_after)}")

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]
            logger.info(f"Samples after max_samples limit: {len(self.samples)}")

        if self.samples:
            first_sample = self.samples[0]
            logger.info("First sample structure:")
            logger.info(f"  emb_tensor shape: {first_sample[0].shape}")
            logger.info(f"  prompt_text: {first_sample[1].encode()}")
            logger.info(f"  target_text: {first_sample[2].encode()}")
            logger.info(f"  task_type: {first_sample[3]}")
            logger.info(f"  sample_id: {first_sample[4]}")
            logger.info(f"  image: {first_sample[5]}")

        # ========== D5.2 / D6.2: Load label CSV if needed ==========
        self.label_lookup: Dict[str, dict] = {}
        self.positive_counts: Dict[str, int] = {}
        if (label_in_prompt or oversample_rare_findings) and label_csv_path:
            label_path = Path(label_csv_path)
            if not label_path.exists():
                raise FileNotFoundError(f"Label CSV not found: {label_path}")
            df_labels = pd.read_csv(label_path)
            for _, row in df_labels.iterrows():
                norm_id = _normalize_image_id(str(row['VolumeName']))
                label_dict = {name: int(row[name]) for name in CTRATE_LABEL_NAMES if name in row}
                self.label_lookup[norm_id] = label_dict
                self.positive_counts[norm_id] = sum(label_dict.values())
            logger.info(f"Loaded {len(self.label_lookup)} label entries from {label_csv_path}")
        elif (label_in_prompt or oversample_rare_findings) and not label_csv_path:
            raise ValueError("label_in_prompt/oversample_rare_findings requires label_csv_path")

        # ========== D5.2: Inject labels into prompts ==========
        self.sample_weights: Optional[List[float]] = None
        if label_in_prompt and self.label_lookup:
            updated = 0
            for i, sample in enumerate(self.samples):
                emb, prompt, target, task_type, sample_id, source_id, has_image, question = sample
                norm_id = _normalize_image_id(str(source_id))
                if norm_id in self.label_lookup and has_image:
                    label_text = _format_labels_as_text(self.label_lookup[norm_id])
                    token = TASK_TOKENS.get(task_type, '')
                    if self.use_chat_template:
                        user_content = f"{label_text}\n{ORGAN_TOKENS_DESC}\n{token}\n{question}"
                        new_prompt, new_target = _build_chat_prompt(self.system_prompt, user_content, gpt_value := target.replace('<|eot_id|>', '').rstrip())
                        self.samples[i] = (emb, new_prompt, new_target, task_type, sample_id, source_id, has_image, question)
                    else:
                        new_prompt = f"{label_text}\n{ORGAN_TOKENS_DESC}\n{token}\n{question}"
                        self.samples[i] = (emb, new_prompt, target, task_type, sample_id, source_id, has_image, question)
                    updated += 1
            logger.info(f"D5.2 Label-in-Prompt: updated {updated}/{len(self.samples)} samples")

        # ========== D6.2: Compute per-sample weights for WeightedRandomSampler ==========
        if oversample_rare_findings and self.positive_counts:
            weights = []
            rare_count = 0
            for sample in self.samples:
                source_id = sample[5]
                norm_id = _normalize_image_id(str(source_id))
                n_pos = self.positive_counts.get(norm_id, 0)
                if n_pos >= oversample_min_positives:
                    weights.append(oversample_factor)
                    rare_count += 1
                else:
                    weights.append(1.0)
            self.sample_weights = weights
            logger.info(f"D6.2 Oversample: {rare_count}/{len(self.samples)} rare samples (>={oversample_min_positives} positives) weighted x{oversample_factor}")

        logger.info("========== ViSDVQADataset.__init__ END ==========")

    def __len__(self) -> int:
        return len(self.samples)

    def get_indices_by_task(self) -> Dict[str, List[int]]:
        """Return indices grouped by task_type for stratified validation sampling."""
        by_task: Dict[str, List[int]] = {}
        for idx, sample in enumerate(self.samples):
            task_type = sample[3]  # task_type is at index 3
            if task_type not in by_task:
                by_task[task_type] = []
            by_task[task_type].append(idx)
        return by_task

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if len(sample) == 9:
            emb, prompt, target, task_type, sample_id, source_id, has_image, question, raw_target = sample
        else:
            emb, prompt, target, task_type, sample_id, source_id, has_image, question = sample
            raw_target = target
        return {
            "embeddings": emb,
            "prompt": prompt,
            "target": target,
            "task_type": task_type,
            "sample_id": sample_id,
            "image": source_id,  # Keep 'image' key for compatibility
            "has_image": has_image,  # Add has_image flag
            "question": question,  # Add question field directly
            "raw_target": raw_target,  # Original text for eval (empty str if organ had no annotation)
        }


def collate_fn_vqa(batch: List[dict]) -> dict:
    """Stack embeddings and keep prompt/target/task_type/sample_id/image/has_image/question as lists."""
    embeddings = torch.stack([b["embeddings"] for b in batch], dim=0)
    prompts = [b["prompt"] for b in batch]
    targets = [b["target"] for b in batch]
    task_types = [b["task_type"] for b in batch]
    sample_ids = [b["sample_id"] for b in batch]
    images = [b["image"] for b in batch]
    has_images = [b["has_image"] for b in batch]
    questions = [b["question"] for b in batch]
    raw_targets = [b.get("raw_target", b["target"]) for b in batch]
    logger = get_module_logger()
    # logger.info("========== collate_fn_vqa ==========")
    # logger.info(f"Batch size: {len(batch)}")
    # logger.info(f"Stacked embeddings shape: {embeddings.shape}")
    # logger.info(f"task_types in batch: {task_types}")
    # logger.info(f"sample_ids in batch: {sample_ids}")
    # logger.info(f"has_images in batch: {has_images}")
    # logger.info(f"First prompt: {trunc_str(prompts[0], 150)}")
    # logger.info(f"First target: {trunc_str(targets[0], 150)}")
    # logger.info(f"First question: {trunc_str(questions[0], 150)}")
    return {
        "embeddings": embeddings,
        "prompt": prompts,
        "target": targets,
        "task_type": task_types,
        "sample_id": sample_ids,
        "image": images,
        "has_image": has_images,
        "question": questions,
        "raw_target": raw_targets,
    }
