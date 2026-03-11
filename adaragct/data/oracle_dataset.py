"""
Step 2.1: OracleTrainDataset — Unified dataset for oracle training.

Config-driven dataset that supports:
- 2 base model types: "raw" (full report) vs "organ" (per-organ generation)
- 3 trigger strategies: "entropy" / "c5_head" / "rag_token"
- 1 selection policy: perplexity-based (from precomputed oracle_contexts.json)

Reads:
- train_organ_reports.json (raw_report + organ_reports per sample)
- oracle_contexts.json (precomputed: which sentences need RAG + retrieved context)
- Embeddings NPZ (ViSD-Boost + optional CT-CLIP for whole_ct)

Special tokens:
- <|ret_start|>, <|ret_end|>: wrap injected context (loss masked)
- [RAG]: trigger token for Strategy C (loss NOT masked, model learns to generate it)
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from adaragct.data.dataset import (
    _normalize_image_id,
    _build_chat_prompt,
    ORGAN_TOKENS_DESC,
    ORGAN_ORDER,
    SYSTEM_PROMPTS,
    TASK_TOKENS,
    DEFAULT_NEGATIVES,
    ORGAN_SYSTEM_PROMPT,
)
from adaragct.utils.logger import get_module_logger
from adaragct.utils.rag_utils import RET_START_TOKEN, RET_END_TOKEN, RAG_TOKEN, is_valid_sentence


class OracleTrainDataset(Dataset):
    """
    Unified dataset for oracle training.

    Config keys:
        base_model_type: "raw" | "organ"
        trigger_strategy: "entropy" | "c5_head" | "rag_token"
        no_context_drop_rate: float (probability of dropping context for needs_rag sentences)
    """

    def __init__(
        self,
        organ_reports_json: str,
        oracle_contexts_json: str,
        embeddings_npz: str,
        embedding_keys: List[str],
        base_model_type: str = "raw",
        trigger_strategy: str = "entropy",
        use_chat_template: bool = True,
        system_prompt: str = "detailed",
        no_context_drop_rate: float = 0.1,
        whole_ct_npz: Optional[str] = None,
        whole_ct_key: Optional[str] = None,
        whole_ct_dim: Optional[int] = None,
        max_samples: Optional[int] = None,
    ):
        logger = get_module_logger()
        logger.info(f"OracleTrainDataset init: base_model_type={base_model_type}, "
                     f"trigger_strategy={trigger_strategy}")

        assert base_model_type in ("raw", "organ"), f"Invalid base_model_type: {base_model_type}"
        assert trigger_strategy in ("entropy", "c5_head", "rag_token"), \
            f"Invalid trigger_strategy: {trigger_strategy}"

        self.base_model_type = base_model_type
        self.trigger_strategy = trigger_strategy
        self.use_chat_template = use_chat_template
        self.no_context_drop_rate = no_context_drop_rate
        self.embedding_keys = list(embedding_keys)

        # Resolve system prompt
        self.system_prompt = SYSTEM_PROMPTS.get(system_prompt, system_prompt)

        # Load organ reports
        logger.info(f"Loading organ reports: {organ_reports_json}")
        with open(organ_reports_json) as f:
            self.organ_reports_list = json.load(f)
        logger.info(f"  {len(self.organ_reports_list)} samples")

        # Load oracle contexts
        logger.info(f"Loading oracle contexts: {oracle_contexts_json}")
        with open(oracle_contexts_json) as f:
            self.oracle_contexts = json.load(f)
        logger.info(f"  {len(self.oracle_contexts)} samples with contexts")

        # Load embeddings
        logger.info(f"Loading embeddings: {embeddings_npz}")
        data = np.load(embeddings_npz, allow_pickle=True)
        sample_ids = data["sample_ids"]
        if hasattr(sample_ids, "tolist"):
            sample_ids = sample_ids.tolist()

        organ_keys = [k for k in embedding_keys if k != embedding_keys[0]]
        wct_key_name = embedding_keys[0]

        # Load organ embeddings from main NPZ
        organ_embs = {k: torch.from_numpy(np.asarray(data[k])).float() for k in organ_keys}
        organ_dim = organ_embs[organ_keys[0]].shape[1]

        # Load whole_ct (may be from separate NPZ)
        if whole_ct_npz is not None:
            wct_data = np.load(whole_ct_npz, allow_pickle=True)
            wct_ids = wct_data["sample_ids"]
            if hasattr(wct_ids, "tolist"):
                wct_ids = wct_ids.tolist()
            self.wct_id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(wct_ids)}
            actual_wct_key = whole_ct_key or wct_key_name
            self.wct_emb = torch.from_numpy(np.asarray(wct_data[actual_wct_key])).float()
            wct_dim = self.wct_emb.shape[1]
        else:
            self.wct_id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(sample_ids)}
            self.wct_emb = torch.from_numpy(np.asarray(data[wct_key_name])).float()
            wct_dim = self.wct_emb.shape[1]

        self.max_dim = max(organ_dim, wct_dim)
        self.organ_embs = organ_embs
        self.organ_keys = organ_keys
        self.id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(sample_ids)}

        # Build sample list (filter to samples with embeddings)
        self.samples = []
        missing = 0
        for item in self.organ_reports_list:
            norm_id = _normalize_image_id(str(item["image"]))
            if norm_id not in self.id_to_idx:
                missing += 1
                continue
            self.samples.append(item)

        if missing > 0:
            logger.info(f"  Skipped {missing} samples without embeddings")

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        logger.info(f"  Final: {len(self.samples)} samples")

        # For organ mode: expand to per-organ samples
        if base_model_type == "organ":
            self._build_organ_samples()
        else:
            self._build_raw_samples()

        logger.info(f"  Total training items: {len(self.items)}")

    def _get_embedding(self, image_id: str) -> torch.Tensor:
        """Build [5, max_dim] embedding tensor."""
        norm_id = _normalize_image_id(str(image_id))
        npz_idx = self.id_to_idx[norm_id]

        emb_list = []
        # whole_ct
        wct_idx = self.wct_id_to_idx.get(norm_id, None)
        if wct_idx is not None:
            wct_vec = self.wct_emb[wct_idx]
            if wct_vec.shape[0] < self.max_dim:
                wct_vec = torch.nn.functional.pad(wct_vec, (0, self.max_dim - wct_vec.shape[0]))
            emb_list.append(wct_vec)
        else:
            emb_list.append(torch.zeros(self.max_dim))

        # organs
        for k in self.organ_keys:
            vec = self.organ_embs[k][npz_idx]
            if vec.shape[0] < self.max_dim:
                vec = torch.nn.functional.pad(vec, (0, self.max_dim - vec.shape[0]))
            emb_list.append(vec)

        return torch.stack(emb_list, dim=0)

    def _get_oracle_contexts_for_sample(self, sample_id: str) -> dict:
        """Get oracle contexts for a sample, organized by organ.

        Returns: {organ: {sentence_idx: context_text}}
        """
        contexts = self.oracle_contexts.get(sample_id, {})
        result = {}
        for organ, ctx_list in contexts.items():
            result[organ] = {}
            for ctx in ctx_list:
                result[organ][ctx["sentence_idx"]] = ctx["retrieved_sentence"]
        return result

    def _inject_context_into_text(
        self,
        organ_text: str,
        oracle_contexts: dict,
        organ: str,
    ) -> Tuple[str, List[dict]]:
        """Inject oracle context into organ text at sentence boundaries.

        Args:
            organ_text: original organ paragraph text
            oracle_contexts: {organ: {sentence_idx: context_text}}
            organ: current organ name

        Returns:
            injected_text: text with context injected
            c5_labels: list of {position: int, needs_rag: bool} for C5 head training
        """
        sentences = sent_tokenize(organ_text)
        sentences = [s for s in sentences if is_valid_sentence(s)]

        if not sentences:
            return organ_text, []

        organ_ctx = oracle_contexts.get(organ, {})
        parts = []
        c5_labels = []

        for sent_idx, sent_text in enumerate(sentences):
            needs_rag = sent_idx in organ_ctx

            # Record C5 label at sentence boundary (before this sentence)
            c5_labels.append({"sentence_idx": sent_idx, "needs_rag": needs_rag})

            if needs_rag:
                # Optionally drop context for robustness
                if random.random() < self.no_context_drop_rate:
                    parts.append(sent_text)
                    continue

                ctx_text = organ_ctx[sent_idx]

                if self.trigger_strategy == "rag_token":
                    # Strategy C: [RAG] token before context
                    parts.append(f"{RAG_TOKEN}{RET_START_TOKEN}{ctx_text}{RET_END_TOKEN} {sent_text}")
                else:
                    # Strategy A/B: just context
                    parts.append(f"{RET_START_TOKEN}{ctx_text}{RET_END_TOKEN} {sent_text}")
            else:
                parts.append(sent_text)

        injected_text = " ".join(parts)
        return injected_text, c5_labels

    def _build_raw_samples(self):
        """Build items for raw model: full report with injected contexts."""
        self.items = []

        for item in self.samples:
            norm_id = _normalize_image_id(str(item["image"]))
            raw_report = item["raw_report"].strip()
            organ_reports = item["organ_reports"]

            assert raw_report, f"Empty raw_report for {norm_id}"

            oracle_ctx = self._get_oracle_contexts_for_sample(norm_id)

            # Inject context into raw_report by processing each organ paragraph
            # and reconstructing the full report
            injected_parts = []
            all_c5_labels = []

            injected_target = raw_report
            for organ in ORGAN_ORDER:
                organ_text = organ_reports.get(organ, "").strip()
                if not organ_text:
                    continue
                injected, c5_labels = self._inject_context_into_text(
                    organ_text, oracle_ctx, organ
                )
                all_c5_labels.extend([{**lbl, "organ": organ} for lbl in c5_labels])
                injected_target = injected_target.replace(organ_text, injected, 1)

            # Build prompt (same as E29 training)
            task_token = TASK_TOKENS["report_generation"]
            user_content = (
                f"{ORGAN_TOKENS_DESC}\n"
                f"{task_token}\n"
                f"Would you mind generating the radiology report for the specified chest CT scan?"
            )

            if self.use_chat_template:
                prompt, target = _build_chat_prompt(
                    self.system_prompt, user_content, injected_target
                )
            else:
                prompt = user_content
                target = injected_target

            self.items.append({
                "sample_id": norm_id,
                "image_id": str(item["image"]),
                "prompt": prompt,
                "target": target,
                "c5_labels": all_c5_labels,
                "task_type": "report_generation",
            })

    def _build_organ_samples(self):
        """Build items for organ model: per-organ with injected contexts."""
        self.items = []

        for item in self.samples:
            norm_id = _normalize_image_id(str(item["image"]))
            organ_reports = item["organ_reports"]
            oracle_ctx = self._get_oracle_contexts_for_sample(norm_id)

            for organ_idx, target_organ in enumerate(ORGAN_ORDER):
                organ_text = organ_reports.get(target_organ, "").strip()
                train_target = organ_text or DEFAULT_NEGATIVES[target_organ]

                # Inject context
                injected_target, c5_labels = self._inject_context_into_text(
                    train_target, oracle_ctx, target_organ
                )

                # Build context from previous organs (GT)
                context_parts = []
                for prev_organ in ORGAN_ORDER[:organ_idx]:
                    prev_text = organ_reports.get(prev_organ, "").strip()
                    filled = prev_text or DEFAULT_NEGATIVES[prev_organ]
                    context_parts.append(f"[{prev_organ.upper()}] {filled}")
                context_str = "\n".join(context_parts)

                # Build prompt
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

                if self.use_chat_template:
                    prompt, target = _build_chat_prompt(
                        self.system_prompt, user_body, injected_target
                    )
                else:
                    prompt = user_body
                    target = injected_target

                self.items.append({
                    "sample_id": f"{norm_id}_{target_organ}",
                    "image_id": str(item["image"]),
                    "prompt": prompt,
                    "target": target,
                    "c5_labels": [
                        {**lbl, "organ": target_organ} for lbl in c5_labels
                    ],
                    "task_type": "report_generation",
                    "organ": target_organ,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        emb = self._get_embedding(item["image_id"])

        return {
            "embeddings": emb,
            "prompt": item["prompt"],
            "target": item["target"],
            "task_type": item["task_type"],
            "sample_id": item["sample_id"],
            "image": item["image_id"],
            "has_image": True,
            "c5_labels": item["c5_labels"],
        }


def collate_fn_oracle(batch: List[dict]) -> dict:
    """Collate function for OracleTrainDataset."""
    return {
        "embeddings": torch.stack([b["embeddings"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "target": [b["target"] for b in batch],
        "task_type": [b["task_type"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
        "image": [b["image"] for b in batch],
        "has_image": [b["has_image"] for b in batch],
        "c5_labels": [b["c5_labels"] for b in batch],
    }
