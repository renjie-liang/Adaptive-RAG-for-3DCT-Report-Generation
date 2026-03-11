"""
Step 1.1: Pre-compute per-sentence perplexity using E29 base model.

Approach:
1. Load E29 checkpoint + tokenizer
2. For each training sample, construct the SAME prompt E29 was trained with
3. Forward full raw_report as target (teacher-forcing)
4. Extract per-token cross-entropy loss
5. Use organ_reports to find organ paragraph boundaries in raw_report
6. Split each organ paragraph into sentences (NLTK)
7. Aggregate token-level loss per sentence → per-sentence perplexity
8. Mark top-X% highest perplexity sentences as needs_rag=True

Usage:
    python P2_rag/oracle/precompute_perplexity.py
    python P2_rag/oracle/precompute_perplexity.py --max-samples 1000
    python P2_rag/oracle/precompute_perplexity.py --ppl-percentile 70

Requires GPU.
"""

import argparse
import json
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from adaragct.models.build_model import build_llava_model_with_lora
from adaragct.train.train_step import project_embeddings
from adaragct.utils.tokenizer_utils import tokenizer_organ_token
from adaragct.constants import ORGAN_TOKEN_NAMES
from adaragct.data.dataset import (
    ORGAN_TOKENS_DESC,
    SYSTEM_PROMPTS,
    ORGAN_ORDER,
    _build_chat_prompt,
    _normalize_image_id,
    TASK_TOKENS,
)
from adaragct.utils.path_utils import load_config_from_checkpoint, apply_default_values

from llava.constants import IGNORE_INDEX

# Organ reports JSON paths
TRAIN_ORGAN_REPORTS_JSON = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/vqa/organ_report_generation/train_organ_reports.json"

# Sentence database (already built in Week 1)
DATABASE_DIR = "results/P2_rag/database"


def is_valid_sentence(s: str) -> bool:
    """Filter out truly empty/meaningless fragments."""
    s = s.strip()
    if not s:
        return False
    if all(c in ' \t\n.,;:!?' for c in s):
        return False
    if s.rstrip(':').lower() in ('lung', 'heart', 'esophagus', 'aorta', 'other', 'findings'):
        return False
    return True


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    """Load E29 model from checkpoint config."""
    config = load_config_from_checkpoint(checkpoint_path)
    config = apply_default_values(config)
    config["model"]["use_flash_attn"] = False

    model, tokenizer = build_llava_model_with_lora(
        config=config,
        device=device,
        pretrain_checkpoint=checkpoint_path,
    )
    model.eval()
    model.config.tokenizer_model_max_length = 4096

    return model, tokenizer, config


def load_embeddings(config: dict):
    """Load training embeddings."""
    train_npz_path = config["data"]["train_embeddings_npz"]
    data = np.load(train_npz_path, allow_pickle=True)
    sample_ids = data["sample_ids"]
    if hasattr(sample_ids, "tolist"):
        sample_ids = sample_ids.tolist()

    organ_keys = config["data"]["organ_keys"]
    whole_ct_key = config["data"].get("whole_ct_key", "whole_ct_attention")

    organ_embs = {k: torch.from_numpy(np.asarray(data[k])).float() for k in organ_keys}
    organ_dim = organ_embs[organ_keys[0]].shape[1]

    # Load whole_ct (may be from separate NPZ)
    whole_ct_npz_path = config["data"].get("whole_ct_embeddings_npz_train", None)
    if whole_ct_npz_path:
        wct_data = np.load(whole_ct_npz_path, allow_pickle=True)
        wct_ids = wct_data["sample_ids"]
        if hasattr(wct_ids, "tolist"):
            wct_ids = wct_ids.tolist()
        wct_id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(wct_ids)}
        wct_emb = torch.from_numpy(np.asarray(wct_data[whole_ct_key])).float()
        wct_dim = wct_emb.shape[1]
    else:
        wct_emb = torch.from_numpy(np.asarray(data[whole_ct_key])).float()
        wct_dim = wct_emb.shape[1]
        wct_id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(sample_ids)}

    max_dim = max(organ_dim, wct_dim)
    id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(sample_ids)}

    return {
        "sample_ids": sample_ids,
        "id_to_idx": id_to_idx,
        "organ_embs": organ_embs,
        "organ_keys": organ_keys,
        "wct_emb": wct_emb,
        "wct_id_to_idx": wct_id_to_idx,
        "wct_dim": wct_dim,
        "organ_dim": organ_dim,
        "max_dim": max_dim,
    }


def get_embedding_tensor(emb_data: dict, norm_id: str) -> torch.Tensor:
    """Build [5, max_dim] embedding tensor for a sample."""
    max_dim = emb_data["max_dim"]
    npz_idx = emb_data["id_to_idx"].get(norm_id, None)
    if npz_idx is None:
        return None

    emb_list = []
    # whole_ct
    wct_idx = emb_data["wct_id_to_idx"].get(norm_id, None)
    if wct_idx is not None:
        wct_vec = emb_data["wct_emb"][wct_idx]
        if wct_vec.shape[0] < max_dim:
            wct_vec = torch.nn.functional.pad(wct_vec, (0, max_dim - wct_vec.shape[0]))
        emb_list.append(wct_vec)
    else:
        emb_list.append(torch.zeros(max_dim))

    # organs
    for k in emb_data["organ_keys"]:
        vec = emb_data["organ_embs"][k][npz_idx]
        if vec.shape[0] < max_dim:
            vec = torch.nn.functional.pad(vec, (0, max_dim - vec.shape[0]))
        emb_list.append(vec)

    return torch.stack(emb_list, dim=0)  # [5, max_dim]


def build_prompt(config: dict) -> str:
    """Build the same prompt E29 was trained with (report_generation task)."""
    system_prompt = SYSTEM_PROMPTS.get(
        config["data"].get("system_prompt", "detailed"),
        config["data"].get("system_prompt", "detailed"),
    )
    task_token = TASK_TOKENS["report_generation"]
    user_content = (
        f"{ORGAN_TOKENS_DESC}\n"
        f"{task_token}\n"
        f"Would you mind generating the radiology report for the specified chest CT scan?"
    )
    prompt_text, _ = _build_chat_prompt(system_prompt, user_content, "placeholder")
    return prompt_text


def find_organ_spans_in_report(raw_report: str, organ_reports: dict) -> list:
    """Find character-level spans of each organ paragraph in raw_report.

    Returns list of (organ, start_char, end_char) sorted by start_char.
    Uses fuzzy matching: finds the longest matching substring.
    """
    spans = []
    for organ in ORGAN_ORDER:
        text = organ_reports.get(organ, "").strip()
        if not text:
            continue
        # Try exact match first
        idx = raw_report.find(text)
        if idx >= 0:
            spans.append((organ, idx, idx + len(text)))
            continue
        # Try first 50 chars as anchor
        anchor = text[:min(50, len(text))]
        idx = raw_report.find(anchor)
        if idx >= 0:
            # Estimate end based on original length
            end = min(idx + len(text), len(raw_report))
            spans.append((organ, idx, end))

    spans.sort(key=lambda x: x[1])
    return spans


def map_char_to_token(text: str, token_ids: list, tokenizer) -> list:
    """Map each token to its character span in the original text.

    Returns list of (start_char, end_char) for each token.
    Uses incremental decoding to align tokens to characters.
    """
    char_spans = []
    decoded_so_far = ""
    for i, tok_id in enumerate(token_ids):
        # Decode up to and including this token
        decoded_up_to = tokenizer.decode(token_ids[:i+1], skip_special_tokens=False)
        # The span of this token is the new characters added
        start = len(decoded_so_far)
        end = len(decoded_up_to)
        char_spans.append((start, end))
        decoded_so_far = decoded_up_to
    return char_spans


@torch.no_grad()
def compute_per_token_loss(
    model: nn.Module,
    tokenizer,
    config: dict,
    emb_tensor: torch.Tensor,
    raw_report: str,
    device: torch.device,
) -> tuple:
    """Compute per-token cross-entropy loss for a single sample.

    Returns:
        target_ids: list of token IDs for the target (raw_report)
        per_token_loss: numpy array of per-token CE loss
    """
    prompt_text = build_prompt(config)
    # Tokenize
    prompt_ids = tokenizer_organ_token(
        prompt=prompt_text,
        tokenizer=tokenizer,
        add_special_tokens=True,
    )

    # Target: "Findings: ..." + <|eot_id|>
    target_text = f"{raw_report}<|eot_id|>"
    target_ids = tokenizer_organ_token(
        prompt=target_text,
        tokenizer=tokenizer,
        add_special_tokens=False,
    )
    if target_ids and target_ids[0] == tokenizer.bos_token_id:
        target_ids = target_ids[1:]

    full_ids = prompt_ids + target_ids
    prompt_len = len(prompt_ids)

    # Truncate
    max_len = 2048
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
        target_len_actual = len(full_ids) - prompt_len
        target_ids = target_ids[:target_len_actual]

    # Build labels (IGNORE_INDEX for prompt, actual IDs for target)
    labels = [IGNORE_INDEX] * prompt_len + target_ids
    if len(labels) > max_len:
        labels = labels[:max_len]

    # To tensors
    input_ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels_t = torch.tensor([labels], dtype=torch.long, device=device)

    # Project embeddings
    embeddings = emb_tensor.unsqueeze(0)  # [1, 5, dim]
    images_list = project_embeddings(
        embeddings=embeddings,
        model=model,
        has_images=[True],
        device=device,
    )

    # Forward
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids_t,
            labels=labels_t,
            images=images_list,
        )

    # Extract per-token loss
    logits = outputs.logits  # [1, seq_len, vocab]
    shift_logits = logits[0, :-1, :].float()  # [seq_len-1, vocab] — cast to fp32 for cross_entropy
    shift_labels = labels_t[0, 1:]    # [seq_len-1]

    # Compute per-token CE loss (no reduction)
    per_token_ce = F_torch.cross_entropy(
        shift_logits, shift_labels, reduction='none', ignore_index=IGNORE_INDEX
    )  # [seq_len-1]

    # Extract only target tokens (skip prompt)
    # shift_labels[i] corresponds to predicting token at position i+1
    # Target starts at position prompt_len, so in shifted space: prompt_len - 1
    target_start = prompt_len - 1
    target_losses = per_token_ce[target_start:].cpu().numpy()

    return target_ids, target_losses


def build_token_char_map(text: str, token_ids: list, tokenizer) -> list:
    """Build a list of (char_start, char_end) for each token in token_ids,
    relative to `text`. Uses incremental decoding to align tokens to characters.

    Returns list of length len(token_ids), each element is (start, end) char offset.
    """
    char_spans = []
    decoded_so_far = ""
    for i, tok_id in enumerate(token_ids):
        decoded_up_to = tokenizer.decode(token_ids[:i + 1], skip_special_tokens=True)
        start = len(decoded_so_far)
        end = len(decoded_up_to)
        char_spans.append((start, end))
        decoded_so_far = decoded_up_to
    return char_spans


def aggregate_sentence_perplexity(
    raw_report: str,
    organ_reports: dict,
    target_ids: list,
    per_token_loss: np.ndarray,
    tokenizer,
) -> dict:
    """Aggregate per-token loss into per-sentence perplexity by organ.

    Strategy: encode raw_report as a whole to get raw_ids (context-aware BPE).
    Find each sentence's char span in raw_report, map to token span in raw_ids,
    then align raw_ids token span to target_ids (which is a suffix of raw_ids
    after the prompt prefix is removed).

    Returns dict: {organ: [{sentence_idx, text, ppl, n_tokens}, ...]}
    """
    result = {}

    # Encode the full raw_report as a whole (context-aware BPE)
    raw_ids = tokenizer.encode(raw_report, add_special_tokens=False)

    # Build char map for raw_ids: char_spans[i] = (char_start, char_end) for raw_ids[i]
    char_spans = build_token_char_map(raw_report, raw_ids, tokenizer)

    # Find where raw_ids aligns with target_ids.
    # target_ids was built from raw_report + <|eot_id|>, so target_ids[:len(raw_ids)] == raw_ids
    # (modulo possible truncation). Compute offset: target_ids[offset:] = raw_ids[:len(target_ids)-offset]
    # In practice offset=0 since target_text = raw_report + eot, so target_ids starts with raw_ids.
    # Verify alignment on first 5 tokens.
    raw_offset = 0  # index into raw_ids corresponding to target_ids[0]
    if len(raw_ids) >= 5 and len(target_ids) >= 5:
        if raw_ids[:5] != target_ids[:5]:
            # Try to find where raw_ids starts in target_ids (shouldn't happen normally)
            for off in range(min(10, len(target_ids))):
                if target_ids[off:off + 5] == raw_ids[:5]:
                    raw_offset = off
                    break

    for organ in ORGAN_ORDER:
        organ_text = organ_reports.get(organ, "").strip()
        if not organ_text:
            result[organ] = []
            continue

        sentences = sent_tokenize(organ_text)
        sentences = [s for s in sentences if is_valid_sentence(s)]
        if not sentences:
            result[organ] = []
            continue

        organ_sentences = []

        not_found_count = 0
        for sent_idx, sent_text in enumerate(sentences):
            # Find sentence char span in raw_report
            char_start = raw_report.find(sent_text)
            if char_start < 0:
                # Try first 40 chars as anchor
                anchor = sent_text[:40]
                char_start = raw_report.find(anchor)
            if char_start < 0:
                # Try normalized whitespace version
                sent_norm = " ".join(sent_text.split())
                raw_norm = " ".join(raw_report.split())
                norm_start = raw_norm.find(sent_norm)
                if norm_start >= 0:
                    # Approximate char_start in original raw_report
                    char_start = raw_report.find(sent_text[:20].strip())
            if char_start < 0:
                not_found_count += 1
                print(f"  [WARN] sent[{sent_idx}] not found in raw_report, skipping: {sent_text[:60]!r}")
                continue

            char_end = char_start + len(sent_text)

            # Map char span to token span in raw_ids
            tok_start = None
            tok_end = None
            for ti, (cs, ce) in enumerate(char_spans):
                if tok_start is None and ce > char_start:
                    tok_start = ti
                if ce >= char_end:
                    tok_end = ti + 1
                    break
            if tok_end is None:
                tok_end = len(raw_ids)
            if tok_start is None:
                tok_start = 0

            # Map raw_ids token span to target_ids span
            # target_ids[i] corresponds to raw_ids[i + raw_offset] (if no truncation)
            tgt_start = tok_start - raw_offset
            tgt_end = tok_end - raw_offset

            n_tokens = tok_end - tok_start

            assert tgt_start >= 0 and tgt_end <= len(per_token_loss), \
                f"sent[{sent_idx}] out of range: tgt={tgt_start}:{tgt_end}, per_token_loss len={len(per_token_loss)}"

            token_losses = per_token_loss[tgt_start:tgt_end]
            valid_losses = token_losses[token_losses > 0]
            if len(valid_losses) > 0:
                avg_loss = float(np.mean(valid_losses))
                ppl = float(np.exp(avg_loss))
            else:
                avg_loss = 0.0
                ppl = 1.0

            organ_sentences.append({
                "sentence_idx": sent_idx,
                "text": sent_text,
                "ppl": round(ppl, 4),
                "avg_loss": round(avg_loss, 4),
                "n_tokens": n_tokens,
            })

        result[organ] = organ_sentences

    return result


def mark_needs_rag(all_results: dict, percentile: float = 70) -> dict:
    """Mark top-X% highest perplexity sentences as needs_rag=True.

    Args:
        all_results: {sample_id: {organ: [{sentence_idx, text, ppl, ...}, ...]}}
        percentile: percentile threshold (e.g., 70 means top 30% get RAG)

    Returns: updated all_results with needs_rag field added
    """
    # Collect all valid perplexities
    all_ppls = []
    for sample_id, organs in all_results.items():
        for organ, sentences in organs.items():
            for sent in sentences:
                if sent["ppl"] > 0:
                    all_ppls.append(sent["ppl"])

    if not all_ppls:
        print("  WARNING: No valid perplexities found!")
        return all_results

    threshold = np.percentile(all_ppls, percentile)
    print(f"\n  Perplexity statistics:")
    print(f"    Total sentences with valid ppl: {len(all_ppls)}")
    print(f"    Mean ppl: {np.mean(all_ppls):.2f}")
    print(f"    Median ppl: {np.median(all_ppls):.2f}")
    print(f"    P{percentile:.0f} threshold: {threshold:.2f}")
    print(f"    Sentences above threshold (needs_rag=True): {sum(1 for p in all_ppls if p >= threshold)}")

    # Mark sentences
    for sample_id, organs in all_results.items():
        for organ, sentences in organs.items():
            for sent in sentences:
                if sent["ppl"] > 0:
                    sent["needs_rag"] = bool(sent["ppl"] >= threshold)
                else:
                    sent["needs_rag"] = False

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Step 1.1: Pre-compute per-sentence perplexity")
    parser.add_argument("--checkpoint", type=str,
                        default="results/base_model/E29_sft_lr2e5/checkpoints/step_4250.pt")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process (None = all)")
    parser.add_argument("--ppl-percentile", type=float, default=70.0,
                        help="Percentile threshold for needs_rag (default: 70 = top 30%%)")
    parser.add_argument("--output-dir", type=str, default="results/P2_rag/oracle")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"\n{'='*70}")
    print(f"Loading E29 model from: {args.checkpoint}")
    print(f"{'='*70}")
    model, tokenizer, config = load_model_and_tokenizer(args.checkpoint, device)

    # Load embeddings
    print(f"\n{'='*70}")
    print(f"Loading training embeddings")
    print(f"{'='*70}")
    emb_data = load_embeddings(config)

    # Load organ reports
    print(f"\nLoading organ reports: {TRAIN_ORGAN_REPORTS_JSON}")
    with open(TRAIN_ORGAN_REPORTS_JSON) as f:
        organ_reports_list = json.load(f)
    print(f"  {len(organ_reports_list)} samples")

    if args.max_samples:
        organ_reports_list = organ_reports_list[:args.max_samples]
        print(f"  Limited to {args.max_samples} samples")

    # Process each sample
    print(f"\n{'='*70}")
    print(f"Computing per-sentence perplexity")
    print(f"{'='*70}")

    all_results = {}
    skipped = 0
    total_sentences = 0
    total_tokens = 0

    for item in tqdm(organ_reports_list, desc="Perplexity"):
        image_id = str(item["image"])
        norm_id = _normalize_image_id(image_id)
        raw_report = item["raw_report"].strip()
        organ_reports = item["organ_reports"]

        # Skip samples with missing embeddings (742 out of 47149 are missing)
        emb_tensor = get_embedding_tensor(emb_data, norm_id)
        if emb_tensor is None:
            skipped += 1
            continue

        # Compute per-token loss
        target_ids, per_token_loss = compute_per_token_loss(
            model, tokenizer, config, emb_tensor, raw_report, device
        )

        # Aggregate into per-sentence perplexity
        sample_result = aggregate_sentence_perplexity(
            raw_report, organ_reports, target_ids, per_token_loss, tokenizer
        )

        all_results[norm_id] = sample_result

        # Stats
        for organ, sents in sample_result.items():
            total_sentences += len(sents)
            total_tokens += sum(s["n_tokens"] for s in sents)

    print(f"\n  Processed: {len(all_results)} samples (skipped {skipped})")
    print(f"  Total sentences: {total_sentences}")
    print(f"  Total tokens: {total_tokens}")

    # Mark needs_rag
    all_results = mark_needs_rag(all_results, percentile=args.ppl_percentile)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sentence_perplexity.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Print per-organ summary
    print(f"\n{'='*70}")
    print(f"PER-ORGAN SUMMARY")
    print(f"{'='*70}")
    for organ in ORGAN_ORDER:
        organ_ppls = []
        organ_rag = 0
        organ_total = 0
        for sample_id, organs in all_results.items():
            for sent in organs.get(organ, []):
                if sent["ppl"] > 0:
                    organ_ppls.append(sent["ppl"])
                    organ_total += 1
                    if sent.get("needs_rag", False):
                        organ_rag += 1
        if organ_ppls:
            print(f"  {organ:12s}: {organ_total:6d} sentences, "
                  f"mean_ppl={np.mean(organ_ppls):6.2f}, "
                  f"median_ppl={np.median(organ_ppls):6.2f}, "
                  f"needs_rag={organ_rag}/{organ_total} ({organ_rag/organ_total*100:.1f}%)")
        else:
            print(f"  {organ:12s}: no valid sentences")


if __name__ == "__main__":
    main()
