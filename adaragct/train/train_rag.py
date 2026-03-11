"""
RAG Token Training: [RAG] trigger token with mixed oracle/real-RAG context.

Context injection format (mid-generation, in target):
    ... sentence N-1. [RAG]<|ret_start|>context<|ret_end|> sentence N ...

Two context sources controlled by oracle_ratio:
  - oracle   (prob=oracle_ratio):   precomputed top3 from oracle_context_top3.jsonl
  - real_rag (prob=1-oracle_ratio): precomputed top3 from retrieval_context_top3.jsonl

Context format (both sources, precomputed top3 — no online ranking/filtering):
    {sample_id: [{"query_sentence": {"text": ...}, "retrieved": [{"text": ...}, ...]}, ...]}

Input: train_reports_csv (Findings_EN + Impressions_EN = raw_report)

Usage:
    python P2_rag/train/train_rag_token.py --config P2_rag/configs/P03_rag_token_oracle.yaml
    python P2_rag/train/train_rag_token.py --config P2_rag/configs/P04_rag_token_mixed.yaml
"""

import argparse
import csv
import json
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from adaragct.utils.seed import set_seed
from adaragct.utils.logger import get_module_logger, create_logger, set_module_logger

from adaragct.models.build_model import build_model
from adaragct.train.train_step import project_embeddings
from adaragct.utils.tokenizer_utils import tokenizer_organ_token
from adaragct.data.dataset import (
    _normalize_image_id,
    _build_chat_prompt,
    ORGAN_TOKENS_DESC,
    ORGAN_ORDER,
    SYSTEM_PROMPTS,
    TASK_TOKENS,
)

from llava.constants import IGNORE_INDEX

from adaragct.utils.rag_utils import RET_START_TOKEN, RET_END_TOKEN, RAG_TOKEN, is_valid_sentence
from adaragct.train.loss import mask_retrieval_context_batch, count_masked_tokens
from adaragct.train.train_utils import load_config, save_checkpoint

import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_text(t: str) -> str:
    """Normalize sentence text for lookup key."""
    return t.strip().strip('.,;:!?').lower()


def _normalize_id(name: str) -> str:
    """Strip .nii.gz / .nii suffix from volume name."""
    s = str(name).strip()
    s = Path(s).name
    if s.endswith('.nii.gz'):
        s = s[:-7]
    elif s.endswith('.nii'):
        s = s[:-4]
    return s


def load_csv_reports(csv_path: str) -> Dict[str, str]:
    """Load train CSV and return {sample_id: raw_report}.

    raw_report = Findings_EN + ' ' + Impressions_EN  (same as build_multi_label_sentences.py)
    """
    logger = get_module_logger()
    logger.info(f"Loading CSV reports: {csv_path}")
    reports = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = _normalize_id(row.get('VolumeName', ''))
            findings = row.get('Findings_EN', '').strip()
            impressions = row.get('Impressions_EN', '').strip()
            raw_report = f"{findings} {impressions}".strip()
            if image_id and raw_report:
                reports[image_id] = raw_report
    logger.info(f"  {len(reports)} reports loaded from CSV")
    return reports


def build_context_index(jsonl_path: str) -> dict:
    """Load precomputed context jsonl and build {(sample_id, norm_text): [text, ...]} index.

    Format: {sample_id: [{"query_sentence": {"text": ...}, "retrieved": [{"text": ...}, ...]}, ...]}
    Order is preserved as-is (ranking already applied offline).
    """
    logger = get_module_logger()
    logger.info(f"  Loading precomputed context: {jsonl_path}")
    with open(jsonl_path) as f:
        data = json.load(f)
    index = {}
    for sample_id, groups in data.items():
        for group in groups:
            qs = group.get("query_sentence", {})
            norm_text = _normalize_text(qs.get("text", ""))
            texts = [e["text"] for e in group.get("retrieved", []) if "text" in e]
            if texts:
                index[(sample_id, norm_text)] = texts
    logger.info(f"    Index: {len(index)} (sample_id, sentence) pairs from {len(data)} samples")
    return index


# ── Dataset ───────────────────────────────────────────────────────────────────

def _extract_patient_id(image_id: str) -> str:
    """Extract patient ID from image_id for self-exclusion.
    e.g. 'train_6002_a_1' → '6002', 'train_14282_a_2' → '14282'
    """
    parts = image_id.split('_')
    if len(parts) >= 2:
        return parts[1]
    return image_id


class RagTokenDataset(Dataset):
    """Dataset for [RAG] token training with mixed oracle / real-RAG context.

    Per-sample context selection (one roll per sample):
        r < no_context_rate              → no context (model learns to generate independently)
        r < no_context_rate+oracle_ratio → oracle context (precomputed top3)
        else                             → real RAG context (precomputed top3)

    The three rates must satisfy: no_context_rate + oracle_ratio + real_rag_ratio == 1.0
    (real_rag_ratio is inferred as 1 - no_context_rate - oracle_ratio)

    Context injection format (in target text):
        [RAG]<|ret_start|>retrieved sentence<|ret_end|> target sentence

    Global context (optional):
        When global_context_json is provided, top-K whole_ct similar reports are
        prepended to the prompt as [RET_START] Reference context: ... [RET_END].
        Same-patient images are excluded from retrieval.

    Retrieval lookup: O(1) via (sample_id, normalized_sentence_text) -> precomputed text list.
    """

    def __init__(
        self,
        train_reports_csv: str,
        oracle_context_jsonl: str,
        embeddings_npz: str,
        embedding_keys: List[str],
        use_chat_template: bool = True,
        system_prompt: str = "detailed",
        no_context_rate: float = 0.1,
        oracle_ratio: float = 0.9,
        max_rag_per_sample: int = -1,
        real_context_jsonl: Optional[str] = None,
        whole_ct_npz: Optional[str] = None,
        whole_ct_key: Optional[str] = None,
        whole_ct_dim: Optional[int] = None,
        max_samples: Optional[int] = None,
        global_context_json: Optional[str] = None,
        sentence_db_json: Optional[str] = None,
        global_context_topk: int = 3,
        global_context_drop_rate: float = 0.1,
    ):
        logger = get_module_logger()
        real_rag_ratio = 1.0 - no_context_rate - oracle_ratio
        assert abs(no_context_rate + oracle_ratio + real_rag_ratio - 1.0) < 1e-6, (
            f"no_context_rate({no_context_rate}) + oracle_ratio({oracle_ratio}) + "
            f"real_rag_ratio({real_rag_ratio:.4f}) must sum to 1.0"
        )
        assert no_context_rate >= 0.0 and oracle_ratio >= 0.0 and real_rag_ratio >= 0.0, (
            "All rates must be non-negative."
        )

        self.use_chat_template = use_chat_template
        self.no_context_rate = no_context_rate
        self.oracle_ratio = oracle_ratio
        self.real_rag_ratio = real_rag_ratio
        self.max_rag_per_sample = max_rag_per_sample  # -1 = no cap
        self.embedding_keys = list(embedding_keys)
        self.system_prompt = SYSTEM_PROMPTS[system_prompt]

        logger.info(f"RagTokenDataset: "
                    f"no_context={no_context_rate:.2f}, oracle={oracle_ratio:.2f}, "
                    f"real_rag={real_rag_ratio:.2f}, "
                    f"max_rag_per_sample={max_rag_per_sample}")

        # Load CSV reports: {sample_id: raw_report}
        self.reports = load_csv_reports(train_reports_csv)

        # Load precomputed oracle context
        self.oracle_index = build_context_index(oracle_context_jsonl)

        # Load precomputed real RAG context (required when real_rag_ratio > 0)
        self.real_rag_index: dict = {}
        if real_rag_ratio > 0.0:
            if real_context_jsonl and os.path.exists(real_context_jsonl):
                self.real_rag_index = build_context_index(real_context_jsonl)
            else:
                raise ValueError(
                    f"real_rag_ratio={real_rag_ratio:.2f} > 0 but real_context_jsonl not found: {real_context_jsonl}"
                )

        data = np.load(embeddings_npz, allow_pickle=True)
        sample_ids = data["sample_ids"]
        if hasattr(sample_ids, "tolist"):
            sample_ids = sample_ids.tolist()

        organ_keys = [k for k in embedding_keys if k != embedding_keys[0]]
        wct_key_name = embedding_keys[0]

        organ_embs = {k: torch.from_numpy(np.asarray(data[k])).float() for k in organ_keys}
        organ_dim = organ_embs[organ_keys[0]].shape[1]

        if whole_ct_npz is not None:
            wct_data = np.load(whole_ct_npz, allow_pickle=True)
            wct_ids = wct_data["sample_ids"]
            if hasattr(wct_ids, "tolist"):
                wct_ids = wct_ids.tolist()
            self.wct_id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(wct_ids)}
            self.wct_emb = torch.from_numpy(np.asarray(wct_data[whole_ct_key or wct_key_name])).float()
        else:
            self.wct_id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(sample_ids)}
            self.wct_emb = torch.from_numpy(np.asarray(data[wct_key_name])).float()

        wct_dim = self.wct_emb.shape[1]
        self.max_dim = max(organ_dim, wct_dim)
        self.organ_embs = organ_embs
        self.organ_keys = organ_keys
        self.id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(sample_ids)}

        # Build sample list: intersection of CSV reports and embeddings
        self.samples = []
        missing_emb = 0
        for norm_id in list(self.reports.keys()):
            if norm_id not in self.id_to_idx:
                missing_emb += 1
                continue
            self.samples.append(norm_id)
        if missing_emb:
            logger.info(f"  Skipped {missing_emb} samples without embeddings")
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        logger.info(f"  Final: {len(self.samples)} samples")

        # ── Global context (hierarchical RAG) ──
        self.global_context_map: Dict[str, str] = {}
        self.global_context_drop_rate = global_context_drop_rate
        if global_context_json and sentence_db_json:
            logger.info(f"  Loading global context: {global_context_json}")
            with open(global_context_json) as f:
                img2img_whole_ct = json.load(f)
            with open(sentence_db_json) as f:
                sentence_db = json.load(f)
            # Build report index: image_id → full report text
            report_index = {img_id: " ".join(s["text"] for s in sents)
                            for img_id, sents in sentence_db.items()}
            # Build per-sample global context with patient-level self-exclusion
            for sid, entries in img2img_whole_ct.items():
                query_patient = _extract_patient_id(sid)
                reports = []
                for entry in entries:
                    cand_id = entry["image_id"]
                    if _extract_patient_id(cand_id) == query_patient:
                        continue
                    r = report_index.get(cand_id, "")
                    if r:
                        reports.append(r)
                    if len(reports) >= global_context_topk:
                        break
                if reports:
                    self.global_context_map[sid] = " ".join(reports)
            logger.info(f"  Global context built for {len(self.global_context_map)} samples "
                        f"(topk={global_context_topk}, drop_rate={global_context_drop_rate})")

    def _get_embedding(self, norm_id: str) -> torch.Tensor:
        npz_idx = self.id_to_idx[norm_id]
        emb_list = []
        wct_idx = self.wct_id_to_idx.get(norm_id)
        if wct_idx is not None:
            v = self.wct_emb[wct_idx]
            if v.shape[0] < self.max_dim:
                v = torch.nn.functional.pad(v, (0, self.max_dim - v.shape[0]))
            emb_list.append(v)
        else:
            emb_list.append(torch.zeros(self.max_dim))
        for k in self.organ_keys:
            v = self.organ_embs[k][npz_idx]
            if v.shape[0] < self.max_dim:
                v = torch.nn.functional.pad(v, (0, self.max_dim - v.shape[0]))
            emb_list.append(v)
        return torch.stack(emb_list, dim=0)

    def _inject_context(self, raw_report: str, sample_id: str, context_mode: str) -> str:
        """Inject [RAG]<|ret_start|>ctx<|ret_end|> before matched sentences in raw_report.

        Steps:
        1. Tokenize raw_report into sentences.
        2. For each sentence, lookup precomputed context via (sample_id, norm_text).
        3. Build injection list (pos, sent, ctx_text).
        4. Cap total injections per sample, then replace from back to front.

        Args:
            context_mode: "no_context" | "oracle" | "real_rag"
        """
        if context_mode == "no_context":
            return raw_report

        index = self.oracle_index if context_mode == "oracle" else self.real_rag_index

        sentences = [s for s in sent_tokenize(raw_report) if is_valid_sentence(s)]
        if not sentences:
            return raw_report

        injections = []   # [(char_pos, sent_text, ctx_text)]
        seen_norms = set()
        for sent in sentences:
            norm = _normalize_text(sent)
            if norm in seen_norms:
                continue
            seen_norms.add(norm)

            texts = index.get((sample_id, norm))
            if not texts:
                continue
            texts = list(texts)
            random.shuffle(texts)
            ctx_text = " ".join(texts)

            # Fuzzy position search: exact, then +'.', then -'.'
            pos = raw_report.find(sent)
            if pos < 0 and not sent.endswith('.'):
                pos = raw_report.find(sent + '.')
            if pos < 0 and sent.endswith('.'):
                pos = raw_report.find(sent[:-1])

            if pos >= 0:
                injections.append((pos, sent, ctx_text))

        if not injections:
            return raw_report

        # Cap total injections per sample
        if self.max_rag_per_sample > 0 and len(injections) > self.max_rag_per_sample:
            injections = random.sample(injections, self.max_rag_per_sample)

        # Sort by position, then replace from back to front (avoid position shift)
        injections.sort(key=lambda x: x[0])
        injected = raw_report
        for pos, sent, ctx_text in reversed(injections):
            inj_str = f"{RAG_TOKEN}{RET_START_TOKEN}{ctx_text}{RET_END_TOKEN} {sent}"
            injected = injected[:pos] + inj_str + injected[pos + len(sent):]

        return injected

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        norm_id = self.samples[idx]
        raw_report = self.reports[norm_id]
        emb = self._get_embedding(norm_id)

        # Three-way context mode selection (one roll per sample)
        r = random.random()
        if r < self.no_context_rate:
            context_mode = "no_context"
        elif r < self.no_context_rate + self.oracle_ratio:
            context_mode = "oracle"
        else:
            context_mode = "real_rag"

        injected = self._inject_context(raw_report, norm_id, context_mode)

        user_content = (
            f"{ORGAN_TOKENS_DESC}\n{TASK_TOKENS['report_generation']}\n"
            "Would you mind generating the radiology report for the specified chest CT scan?"
        )
        if self.use_chat_template:
            prompt, target = _build_chat_prompt(self.system_prompt, user_content, injected)
        else:
            prompt, target = user_content, injected

        # Prepend global context (if available and not dropped)
        has_gc = False
        if self.global_context_map and random.random() >= self.global_context_drop_rate:
            gc_text = self.global_context_map.get(norm_id)
            if gc_text:
                prompt = f"{RET_START_TOKEN} Reference context: {gc_text} {RET_END_TOKEN}\n" + prompt
                has_gc = True

        return {
            "embeddings": emb,
            "prompt": prompt,
            "target": target,
            "has_image": True,
            "sample_id": norm_id,
            "context_mode": context_mode,
            "has_global_context": has_gc,
        }

def collate_fn_rag_token(batch: List[dict]) -> dict:
    return {
        "embeddings": torch.stack([b["embeddings"] for b in batch], dim=0),
        "prompt": [b["prompt"] for b in batch],
        "target": [b["target"] for b in batch],
        "has_image": [b["has_image"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
        "context_mode": [b["context_mode"] for b in batch],
        "has_global_context": [b["has_global_context"] for b in batch],
    }


# ── Model setup ───────────────────────────────────────────────────────────────

def build_model_and_tokenizer(config: dict, device: torch.device):
    logger = get_module_logger()

    # Standard build_model: loads merged base (E41_merged) + projector + fresh LoRA
    model, tokenizer = build_model(
        config=config,
        device=device,
        pretrain_checkpoint=config["model"]["pretrain_checkpoint"],
    )

    # Add RAG tokens (not in the merged base tokenizer)
    rag_tokens = [RET_START_TOKEN, RET_END_TOKEN, RAG_TOKEN]
    new_tokens = [t for t in rag_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        logger.info(f"Added {len(new_tokens)} RAG tokens: {new_tokens}")
        logger.info(f"Final vocab size: {len(tokenizer)}")

    ret_start_id = tokenizer.convert_tokens_to_ids(RET_START_TOKEN)
    ret_end_id = tokenizer.convert_tokens_to_ids(RET_END_TOKEN)
    rag_token_id = tokenizer.convert_tokens_to_ids(RAG_TOKEN)
    assert ret_start_id != tokenizer.unk_token_id, f"RET_START_TOKEN not in tokenizer"
    assert ret_end_id != tokenizer.unk_token_id, f"RET_END_TOKEN not in tokenizer"
    assert rag_token_id != tokenizer.unk_token_id, f"RAG_TOKEN not in tokenizer"
    logger.info(f"ret_start_id={ret_start_id}, ret_end_id={ret_end_id}, rag_token_id={rag_token_id}")

    return model, tokenizer, ret_start_id, ret_end_id


# ── Train step ────────────────────────────────────────────────────────────────

def rag_token_train_step(
    model: nn.Module,
    batch: Dict[str, Any],
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    step: int,
    config: dict,
    ret_start_id: int,
    ret_end_id: int,
    should_print: bool = False,
) -> Tuple[float, Dict[str, float]]:
    model.train()

    use_chat_template = config["data"]["use_chat_template"]
    max_seq_length = config["training"]["max_seq_length_by_task"]["report_generation"]
    grad_accum_steps = config["training"]["grad_accum_steps"]

    embeddings = batch["embeddings"].to(device)
    batch_size = embeddings.shape[0]

    full_ids_list, labels_list = [], []
    for i in range(batch_size):
        prompt_ids = tokenizer_organ_token(
            prompt=batch["prompt"][i], tokenizer=tokenizer, add_special_tokens=True,
        )
        target_text = batch["target"][i]
        if not use_chat_template:
            target_text = target_text + tokenizer.eos_token
        target_ids = tokenizer_organ_token(
            prompt=target_text, tokenizer=tokenizer, add_special_tokens=False,
        )
        if target_ids and target_ids[0] == tokenizer.bos_token_id:
            target_ids = target_ids[1:]

        full_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
        if len(full_ids) > max_seq_length:
            full_ids = full_ids[:max_seq_length]
            labels = labels[:max_seq_length]
        full_ids_list.append(full_ids)
        labels_list.append(labels)

    max_len = max(len(x) for x in full_ids_list)
    pad_id = tokenizer.pad_token_id
    full_ids_padded, labels_padded, attn_mask = [], [], []
    for fids, labs in zip(full_ids_list, labels_list):
        pad = max_len - len(fids)
        full_ids_padded.append(fids + [pad_id] * pad)
        labels_padded.append(labs + [IGNORE_INDEX] * pad)
        attn_mask.append([1] * len(fids) + [0] * pad)

    full_ids_t = torch.tensor(full_ids_padded, dtype=torch.long, device=device)
    labels_t = torch.tensor(labels_padded, dtype=torch.long, device=device)
    attn_t = torch.tensor(attn_mask, dtype=torch.long, device=device)

    labels_t = mask_retrieval_context_batch(labels_t, full_ids_t, ret_start_id, ret_end_id)
    rag_token_loss_weight = config.get("rag_token", {}).get("rag_token_loss_weight", 1.0)
    rag_token_id = tokenizer.convert_tokens_to_ids(RAG_TOKEN)
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        images_list = project_embeddings(
            embeddings=embeddings, model=model,
            has_images=batch["has_image"], device=device,
        )
        if rag_token_loss_weight > 1.0:
            outputs = model(
                input_ids=full_ids_t, attention_mask=attn_t,
                labels=None, images=images_list,
            )
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels_t[..., 1:].contiguous()
            per_token_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction='none',
            ).view(batch_size, -1)
            rag_mask = (shift_labels == rag_token_id).float()
            weight_map = torch.ones_like(rag_mask) + rag_mask * (rag_token_loss_weight - 1.0)
            active_mask = (shift_labels != IGNORE_INDEX).float()
            weighted_loss = (per_token_loss * weight_map * active_mask).sum() / (weight_map * active_mask).sum().clamp(min=1)
            loss = weighted_loss / grad_accum_steps
        else:
            outputs = model(
                input_ids=full_ids_t, attention_mask=attn_t,
                labels=labels_t, images=images_list,
            )
            loss = outputs.loss / grad_accum_steps

    loss.backward()

    metrics = {"loss": loss.item() * grad_accum_steps}
    if should_print:
        mask_stats = count_masked_tokens(labels_t)
        metrics["mask_ratio"] = mask_stats["mask_ratio"]
        metrics["active_tokens"] = mask_stats["active"]
        metrics["masked_tokens"] = mask_stats["masked"]
        modes = batch["context_mode"]
        metrics["frac_no_ctx"] = sum(1 for m in modes if m == "no_context") / batch_size
        metrics["frac_oracle"] = sum(1 for m in modes if m == "oracle") / batch_size
        metrics["frac_real_rag"] = sum(1 for m in modes if m == "real_rag") / batch_size
        gc_flags = batch.get("has_global_context", [])
        if gc_flags:
            metrics["frac_gc"] = sum(1 for g in gc_flags if g) / batch_size

        # [RAG] trigger rate: avg number of [RAG] tokens per sample in target
        metrics["avg_rag_per_sample"] = (full_ids_t == rag_token_id).float().sum(dim=1).mean().item()
        if rag_token_loss_weight > 1.0:
            metrics["rag_loss_weight"] = rag_token_loss_weight

        # [RAG] token embedding gradient norm (proxy for whether model is learning to use RAG)
        emb_weight = model.get_input_embeddings().weight
        if emb_weight.grad is not None:
            metrics["rag_emb_grad"] = emb_weight.grad[rag_token_id].norm().item()

        # Loss split: no_context samples vs context samples
        no_ctx_indices = [i for i, m in enumerate(modes) if m == "no_context"]
        ctx_indices    = [i for i, m in enumerate(modes) if m != "no_context"]
        if no_ctx_indices and ctx_indices:
            with torch.no_grad():
                per_token_loss = torch.nn.functional.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels_t.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="none",
                ).view(batch_size, -1)
                active = (labels_t != IGNORE_INDEX).float()
                per_sample_loss = (per_token_loss * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)
                metrics["loss_no_ctx"] = per_sample_loss[no_ctx_indices].mean().item()
                metrics["loss_ctx"]    = per_sample_loss[ctx_indices].mean().item()

    is_accum_step = (step + 1) % grad_accum_steps == 0
    if is_accum_step:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        metrics["grad_norm"] = grad_norm.item()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
            metrics["lr"] = scheduler.get_last_lr()[0]

    return loss.item() * grad_accum_steps, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Token Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    logger = create_logger(config)
    set_module_logger(logger)

    set_seed(config["experiment"]["seed"])
    device = torch.device(config["device"]["type"])

    rag_config = config["rag_token"]
    no_context_rate = rag_config["no_context_rate"]
    oracle_ratio = rag_config["oracle_ratio"]
    max_rag_per_sample = rag_config.get("max_rag_per_sample", -1)
    real_rag_ratio = 1.0 - no_context_rate - oracle_ratio

    rag_token_loss_weight = rag_config.get("rag_token_loss_weight", 1.0)

    logger.info(f"{'='*70}")
    logger.info(f"RAG Token Training: {config['experiment']['name']}")
    logger.info(f"no_context={no_context_rate:.2f}, oracle={oracle_ratio:.2f}, "
                f"real_rag={real_rag_ratio:.2f}, max_rag={max_rag_per_sample}")
    if rag_token_loss_weight > 1.0:
        logger.info(f"[RAG] token loss weight: {rag_token_loss_weight:.1f}x (A5 enabled)")
    logger.info(f"{'='*70}")

    model, tokenizer, ret_start_id, ret_end_id = build_model_and_tokenizer(config, device)

    data_config = config["data"]
    gc_config = config.get("global_context", {})
    dataset = RagTokenDataset(
        train_reports_csv=data_config["train_reports_csv"],
        oracle_context_jsonl=data_config["oracle_context_jsonl"],
        embeddings_npz=data_config["train_embeddings_npz"],
        embedding_keys=[data_config["whole_ct_key"]] + data_config["organ_keys"],
        use_chat_template=data_config["use_chat_template"],
        system_prompt=data_config["system_prompt"],
        no_context_rate=no_context_rate,
        oracle_ratio=oracle_ratio,
        max_rag_per_sample=max_rag_per_sample,
        real_context_jsonl=data_config.get("real_context_jsonl"),
        whole_ct_npz=data_config["whole_ct_embeddings_npz_train"],
        whole_ct_key=data_config["whole_ct_key"],
        whole_ct_dim=data_config["whole_ct_dim"],
        global_context_json=gc_config.get("img2img_json"),
        sentence_db_json=gc_config.get("sentence_db_json"),
        global_context_topk=gc_config.get("topk", 3),
        global_context_drop_rate=gc_config.get("drop_rate", 0.1),
    )

    train_config = config["training"]
    dataloader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        collate_fn=collate_fn_rag_token,
        pin_memory=True,
        drop_last=True,
    )

    opt_config = train_config["optimizer"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_config["lr"],
        weight_decay=opt_config["weight_decay"],
        betas=tuple(opt_config["betas"]),
        eps=opt_config["eps"],
    )

    from torch.optim.lr_scheduler import LambdaLR
    max_steps = train_config["max_steps"]
    warmup_steps = train_config["scheduler"]["warmup_steps"]
    min_lr_ratio = train_config["scheduler"]["min_lr_ratio"]

    def warmup_cosine_lr(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    save_every = train_config["save_every"]
    log_every = config["logging"]["log_every"]

    logger.info(f"Dataset: {len(dataset)} items | Steps: {max_steps} | Output: {output_dir}")

    global_step = 0
    running_loss = 0.0

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break

            next_step = global_step + 1
            should_print = (next_step % log_every == 0)
            loss_val, metrics = rag_token_train_step(
                model=model, batch=batch, tokenizer=tokenizer,
                optimizer=optimizer, scheduler=scheduler,
                device=device, step=global_step, config=config,
                ret_start_id=ret_start_id, ret_end_id=ret_end_id,
                should_print=should_print,
            )
            running_loss += metrics["loss"]
            global_step += 1

            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                log_msg = f"Step {global_step}/{max_steps} | loss={avg_loss:.4f}"
                if "lr" in metrics:
                    log_msg += f" | lr={metrics['lr']:.2e}"
                if "grad_norm" in metrics:
                    log_msg += f" | grad_norm={metrics['grad_norm']:.3f}"
                if "mask_ratio" in metrics:
                    log_msg += (f" | mask={metrics['mask_ratio']:.2%}"
                                f" (active={metrics['active_tokens']}, masked={metrics['masked_tokens']})")
                if "frac_oracle" in metrics:
                    log_msg += (f" | no_ctx={metrics['frac_no_ctx']:.2f}"
                                f" oracle={metrics['frac_oracle']:.2f}"
                                f" real_rag={metrics['frac_real_rag']:.2f}")
                if "frac_gc" in metrics:
                    log_msg += f" | gc={metrics['frac_gc']:.2f}"
                if "avg_rag_per_sample" in metrics:
                    log_msg += f" | rag_trigger={metrics['avg_rag_per_sample']:.2f}"
                if "rag_emb_grad" in metrics:
                    log_msg += f" | rag_emb_grad={metrics['rag_emb_grad']:.2e}"
                if "loss_no_ctx" in metrics:
                    log_msg += (f" | loss_no_ctx={metrics['loss_no_ctx']:.4f}"
                                f" loss_ctx={metrics['loss_ctx']:.4f}")
                logger.info(log_msg)
                running_loss = 0.0

            if global_step % save_every == 0 or global_step == 1:
                ckpt_path = save_checkpoint(
                    model, tokenizer, optimizer, scheduler,
                    global_step, metrics, config, output_dir,
                )
                logger.info(f"Saved: {ckpt_path}")

    ckpt_path = save_checkpoint(
        model, tokenizer, optimizer, scheduler,
        global_step, metrics, config, output_dir,
    )
    logger.info(f"Training complete! Final: {ckpt_path}")


if __name__ == "__main__":
    main()
