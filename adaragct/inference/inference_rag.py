"""
RAG Token Inference: token-by-token generation with KV cache + [RAG] trigger.

Model generates freely. When it produces [RAG], we retrieve context and inject it.

Modes:
  (default)       : Normal RAG — FLARE-style probe + online two-stage retrieval
                    Stage 1: precomputed img2img_top20 (image-based coarse)
                    Stage 2: encode probe text → cosine + MMR re-rank → top-3 context
  --oracle        : Oracle RAG — use precomputed oracle context (hardcoded path)
  --no-rag        : disable all injection (baseline ablation)
  --corrupt-rag   : inject random wrong context (TODO)

FLARE-style probe (Normal RAG):
  When [RAG] is generated:
    1. Save KV cache position
    2. Forward [RAG], then greedily decode until sentence end (probe)
    3. Crop KV cache back to saved position (discard probe KVs)
    4. Use probe text as query → online retrieval → get context
    5. Forward [RAG] + context together (same as oracle mode injection)

KV cache design:
  - Phase 1 (prefill): full forward with images → initial KV cache
  - Phase 2 (decode):  token-by-token, reuse KV cache (never reset)
  - Phase 3 (RAG):     probe + crop + retrieve + inject context
  - Phase 4 (forward): forward [RAG] + ctx_ids or single token

Usage:
    # Normal RAG (default)
    python P2_rag/inference/inference_rag_token.py \\
        --checkpoint results/rag_oracle_xxx/P04_rag_token_mixed/checkpoints/step_3000

    # Oracle RAG
    python P2_rag/inference/inference_rag_token.py ... --oracle

    # Ablation: no RAG
    python P2_rag/inference/inference_rag_token.py ... --no-rag
"""

import argparse
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from adaragct.models.build_model_rag import load_model_peft
from adaragct.train.train_step import project_embeddings
from adaragct.utils.tokenizer_utils import tokenizer_organ_token
from adaragct.data.dataset import (
    ORGAN_TOKENS_DESC, ORGAN_ORDER, SYSTEM_PROMPTS,
    _build_chat_prompt, _normalize_image_id, TASK_TOKENS,
)

from adaragct.utils.rag_utils import RET_START_TOKEN, RET_END_TOKEN, RAG_TOKEN
from adaragct.train.train_rag import _normalize_text
from collections import Counter
from transformers import DynamicCache

# ── Context index ──────────────────────────────────────────────────────────────

def build_context_index(jsonl_path: str) -> dict:
    """Load precomputed context and build {(sample_id, norm_text): [text, ...]} index."""
    print(f"Loading context: {jsonl_path}")
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
    print(f"  Index: {len(index)} (sample_id, sentence) pairs from {len(data)} samples")
    return index


def lookup_context(
    index: dict,
    sample_id: str,
    partial_text: str,
    corrupt: bool = False,
    all_texts: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Look up context for the most recent sentence in partial_text.

    Tries each sentence (from last to first) until a match is found.
    If corrupt=True, returns random texts from all_texts instead.
    """
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(partial_text)
    for sent in reversed(sentences):
        norm = _normalize_text(sent)
        key = (sample_id, norm)
        if key in index:
            if corrupt and all_texts:
                return random.sample(all_texts, min(3, len(all_texts)))
            return index[key]
    return None


# ── Normal RAG: online retrieval utilities ─────────────────────────────────────

VISD_BOOST_DIR = "data/retrieval/visd_boost"
E2E_CKPT = "data/retrieval/best_e2e.pth"
DATABASE_DIR = "data/retrieval/database"
IMG2IMG_PATH = "data/retrieval/img2img_top20.json"
IMG2IMG_WHOLE_CT_PATH = "data/retrieval/img2img_top20_whole_ct.json"
SENTENCE_DB_PATH = "data/retrieval/train_sentences_multi_label.json"

ORGAN_KEYWORDS = {
    "lung": ["lung", "pulmonary", "bronch", "lobe", "nodule", "opacity", "atelectasis",
             "pleural", "pneumo", "airway", "trachea", "emphysema", "fibrosis",
             "consolidat", "effusion", "interstitial"],
    "heart": ["heart", "cardiac", "pericardi", "coronary", "ventricl", "atri", "valve",
              "cardiomegaly", "myocard"],
    "esophagus": ["esophag", "gastro", "hiatal", "hernia"],
    "aorta": ["aort", "aneurysm", "dissect", "vascular", "arteri", "calcifi"],
}


def detect_organ(text: str, last_organ: str = "lung") -> str:
    """Detect organ from text using keyword matching. Fallback to last_organ."""
    text_lower = text.lower()
    scores = {}
    for organ, keywords in ORGAN_KEYWORDS.items():
        scores[organ] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return last_organ


def load_text_encoder(device: torch.device):
    """Load E2E fine-tuned ViSD-Boost model for text encoding."""
    print(f"[LOAD] Loading E2E text encoder from {E2E_CKPT}")
    sys.path.insert(0, VISD_BOOST_DIR)
    from lavis.utils.vit import ViT
    from lavis.utils.model import VISDBOOST
    from lavis.utils.med import XBertEncoder

    image_encoder = ViT(
        in_channels=1, img_size=(112, 256, 352),
        patch_size=(16, 16, 32), num_classes=0,
        dropout_rate=0.1, qkv_bias=True
    )
    text_encoder = XBertEncoder.from_config(None, from_pretrained=True)
    e2e_model = VISDBOOST(image_encoder=image_encoder, text_encoder=text_encoder)

    ckpt = torch.load(E2E_CKPT, map_location='cpu')
    e2e_model.load_state_dict(ckpt['model'], strict=True)
    print(f"  Epoch: {ckpt.get('epoch', '?')}, val_loss: {ckpt.get('val_loss', '?')}")

    e2e_model.eval()
    e2e_model.to(device)
    return e2e_model


@torch.inference_mode()
def encode_single_sentence(e2e_model, text: str) -> np.ndarray:
    """Encode a single sentence to 256-dim embedding using E2E text encoder."""
    device = next(e2e_model.parameters()).device
    tokens = e2e_model.tokenizer(
        [text], padding="max_length", truncation=True,
        max_length=e2e_model.max_txt_len, return_tensors="pt",
    ).to(device)
    text_output = e2e_model.text_encoder.forward_text(tokens)
    text_feat = F.normalize(
        e2e_model.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
    )
    return text_feat.cpu().numpy()[0]  # (256,)


def load_sentence_db():
    """Load train sentence texts and precomputed embeddings."""
    print(f"[LOAD] Loading sentence database from {DATABASE_DIR}")
    with open(f"{DATABASE_DIR}/train_sentences_multi_label.json") as f:
        sentence_data = json.load(f)

    npz = np.load(f"{DATABASE_DIR}/train_sentence_embeds.npz", allow_pickle=True)
    sentence_embeds = npz["embeddings"]  # (N, 256) float32
    sentence_ids_arr = npz["sentence_ids"]

    sid_to_idx = {str(sid): i for i, sid in enumerate(sentence_ids_arr)}

    sid_to_sent = {}
    for sample_id, sents in sentence_data.items():
        for s in sents:
            sid_to_sent[s["sentence_id"]] = s

    print(f"  {len(sentence_data)} samples, {len(sentence_embeds)} embeddings, "
          f"{len(sid_to_sent)} sentences indexed")
    return sentence_data, sentence_embeds, sid_to_idx, sid_to_sent


def load_img2img_top20():
    """Load precomputed Stage 1 img2img retrieval results."""
    print(f"[LOAD] Loading img2img_top20 from {IMG2IMG_PATH}")
    with open(IMG2IMG_PATH) as f:
        data = json.load(f)
    print(f"  {len(data)} samples")
    return data


# ── MMR helpers (cosine + BLEU-2, no disease_jaccard) ─────────────────────────

def _ngram_precision(hyp: list, ref: list, n: int) -> float:
    if len(hyp) < n:
        return 0.0
    h = Counter(tuple(hyp[i:i+n]) for i in range(len(hyp) - n + 1))
    r = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
    clipped = sum(min(c, r[g]) for g, c in h.items())
    total = sum(h.values())
    return clipped / total if total > 0 else 0.0


def bleu2(text_a: str, text_b: str) -> float:
    """Fast BLEU-2 (geometric mean of P1 and P2, no brevity penalty)."""
    a = text_a.lower().split()
    b = text_b.lower().split()
    p1 = _ngram_precision(a, b, 1)
    p2 = _ngram_precision(a, b, 2)
    return (p1 * p2) ** 0.5 if p1 > 0 and p2 > 0 else 0.0


def mmr_select_online(candidates: list, top_k: int = 3, mmr_lambda: float = 0.7) -> list:
    """MMR selection: cosine relevance + BLEU-2 diversity. No disease_jaccard."""
    if not candidates or len(candidates) <= top_k:
        return candidates
    selected = []
    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        best_idx, best_score = -1, -1e9
        for i, cand in enumerate(remaining):
            relevance = mmr_lambda * cand["cosine_score"]
            if not selected:
                redundancy = 0.0
            else:
                redundancy = max(bleu2(cand["text"], s["text"]) for s in selected)
            score = relevance - (1.0 - mmr_lambda) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected


def online_retrieve_context(
    probe_text: str,
    sample_id: str,
    generated_text: str,
    e2e_model,
    img2img_top20: dict,
    sentence_data: dict,
    sentence_embeds: np.ndarray,
    sid_to_idx: dict,
    last_organ: str = "lung",
    top_k: int = 3,
    mmr_lambda: float = 0.7,
    retrieval_stage: str = "two",
) -> Tuple[Optional[List[str]], str]:
    """FLARE-style online retrieval.

    Two-stage (default):
      1. Detect organ from generated_text + probe_text
      2. Get top-20 train images from img2img_top20 for this sample+organ
      3. Collect candidate sentences (organ-filtered)
      4. Encode probe_text → cosine similarity → MMR top-k

    One-stage (retrieval_stage="one"):
      1. Detect organ from generated_text + probe_text
      2. Collect ALL sentences in database matching detected organ
      3. Encode probe_text → cosine similarity → MMR top-k
      (Skips img2img Stage 1 entirely — pure text-to-text retrieval)

    Returns: (ctx_texts or None, detected_organ)
    """
    # 1. Detect organ
    full_text = generated_text + " " + probe_text
    organ = detect_organ(full_text, last_organ)

    candidates = []
    seen_sids = set()

    if retrieval_stage == "one":
        # One-stage: search ALL sentences in database, NO organ filtering
        for img_id, sents in sentence_data.items():
            for s in sents:
                sid = s["sentence_id"]
                if sid in seen_sids:
                    continue
                if sid in sid_to_idx:
                    seen_sids.add(sid)
                    candidates.append({
                        "sentence_id": sid,
                        "text": s["text"],
                        "organs": s.get("organs", []),
                        "normal": s.get("normal", False),
                        "embed_idx": sid_to_idx[sid],
                    })
    else:
        # Two-stage: narrow by img2img_top20 first
        if sample_id not in img2img_top20:
            print(f"[RETRIEVE] WARNING: sample_id={sample_id!r} not in img2img_top20")
            return None, organ

        organ_entries = img2img_top20[sample_id].get(organ, [])
        if not organ_entries:
            print(f"[RETRIEVE] WARNING: no img2img entries for organ={organ!r}")
            return None, organ

        train_image_ids = [e["image_id"] for e in organ_entries]

        for img_id in train_image_ids:
            sents = sentence_data.get(img_id, [])
            for s in sents:
                sid = s["sentence_id"]
                if sid in seen_sids:
                    continue
                if organ in s.get("organs", []):
                    if sid in sid_to_idx:
                        seen_sids.add(sid)
                        candidates.append({
                            "sentence_id": sid,
                            "text": s["text"],
                            "organs": s.get("organs", []),
                            "normal": s.get("normal", False),
                            "embed_idx": sid_to_idx[sid],
                        })

    if not candidates:
        if retrieval_stage == "one":
            print(f"[RETRIEVE] WARNING: 0 candidates for organ={organ!r} in full database")
        else:
            print(f"[RETRIEVE] WARNING: 0 candidates for organ={organ!r} "
                  f"from {len(train_image_ids)} images")
        return None, organ

    # 4. Encode probe text → query embedding
    query_embed = encode_single_sentence(e2e_model, probe_text)  # (256,)

    # 5. Cosine similarity
    cand_indices = [c["embed_idx"] for c in candidates]
    cand_embeds = sentence_embeds[cand_indices]  # (N, 256)
    cand_norms = np.linalg.norm(cand_embeds, axis=1, keepdims=True)
    cand_norms = np.maximum(cand_norms, 1e-8)
    cand_embeds_n = cand_embeds / cand_norms
    q_norm = max(np.linalg.norm(query_embed), 1e-8)
    query_n = query_embed / q_norm

    cosines = cand_embeds_n @ query_n  # (N,)
    for i, cand in enumerate(candidates):
        cand["cosine_score"] = float(cosines[i])

    # Sort by cosine, take top-50 for MMR efficiency
    candidates.sort(key=lambda x: x["cosine_score"], reverse=True)
    candidates_for_mmr = candidates[:50]

    # 6. MMR select
    selected = mmr_select_online(candidates_for_mmr, top_k=top_k, mmr_lambda=mmr_lambda)

    if not selected:
        return None, organ

    ctx_texts = [s["text"] for s in selected]
    print(f"[RETRIEVE] organ={organ} | total_candidates={len(candidates)} | "
          f"probe_text={probe_text!r}")
    for i, s in enumerate(selected):
        print(f"  [CTX_{i}] cosine={s['cosine_score']:.4f} | {s['text']}")

    return ctx_texts, organ


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(checkpoint_dir: str, device: torch.device):
    """Load PEFT checkpoint using load_model_peft (same as predict_batch.py).
    Returns model, tokenizer, special token ids.
    """
    model, tokenizer = load_model_peft(
        checkpoint_dir=checkpoint_dir,
        device=device,
        merge_lora=False,   # keep LoRA for inference (same as predict_batch.py)
    )
    model.eval()
    model.config.use_cache = True
    model.config.tokenizer_model_max_length = 8192

    ret_start_id = tokenizer.convert_tokens_to_ids(RET_START_TOKEN)
    ret_end_id   = tokenizer.convert_tokens_to_ids(RET_END_TOKEN)
    rag_token_id = tokenizer.convert_tokens_to_ids(RAG_TOKEN)

    unk_id = tokenizer.unk_token_id
    assert ret_start_id != unk_id and ret_start_id > 0, \
        f"[ASSERT] {RET_START_TOKEN} not in tokenizer vocab (id={ret_start_id})"
    assert ret_end_id != unk_id and ret_end_id > 0, \
        f"[ASSERT] {RET_END_TOKEN} not in tokenizer vocab (id={ret_end_id})"
    assert rag_token_id != unk_id and rag_token_id > 0, \
        f"[ASSERT] {RAG_TOKEN} not in tokenizer vocab (id={rag_token_id})"
    assert ret_start_id != ret_end_id != rag_token_id, \
        f"[ASSERT] special token ids must be distinct: {ret_start_id}, {ret_end_id}, {rag_token_id}"
    assert getattr(model.config, 'projector_type', None) is not None, \
        "[ASSERT] model.config.projector_type not set — load_model_peft may have failed"

    # Verify projector weights are loaded (not random)
    try:
        base = model.get_model() if hasattr(model, 'get_model') else model.model
        if hasattr(base, 'organ_projectors'):
            proj = base.organ_projectors
            sample_key = next(iter(proj.keys()))
            sample_w = next(proj[sample_key].parameters())
            print(f"[PROJ_DEBUG] organ_projectors found | sample_key={sample_key} "
                  f"weight_shape={sample_w.shape} mean={sample_w.float().mean().item():.4f} "
                  f"std={sample_w.float().std().item():.4f}")
        elif hasattr(base, 'mm_projector'):
            sample_w = next(base.mm_projector.parameters())
            print(f"[PROJ_DEBUG] mm_projector found | weight_shape={sample_w.shape} "
                  f"mean={sample_w.float().mean().item():.4f} std={sample_w.float().std().item():.4f}")
        else:
            print(f"[PROJ_DEBUG] WARNING: no projector found on model!")
    except Exception as e:
        print(f"[PROJ_DEBUG] ERROR checking projector: {e}")

    print(f"  ret_start_id={ret_start_id}, ret_end_id={ret_end_id}, rag_token_id={rag_token_id}")
    print(f"  projector_type={getattr(model.config, 'projector_type', 'unknown')}")
    print(f"  tokens_per_embedding={getattr(model.config, 'tokens_per_embedding', 'unknown')}")
    return model, tokenizer, ret_start_id, ret_end_id, rag_token_id


# ── Validation data ────────────────────────────────────────────────────────────

def load_validation_data(config: dict, max_samples: Optional[int] = None) -> List[dict]:
    data_cfg = config["data"]
    organ_keys   = data_cfg["organ_keys"]
    whole_ct_key = data_cfg["whole_ct_key"]

    # Load organ embeddings
    npz = np.load(data_cfg["val_embeddings_npz"], allow_pickle=True)
    sample_ids = npz["sample_ids"].tolist()
    organ_embs = {k: torch.from_numpy(np.asarray(npz[k])).float() for k in organ_keys}
    organ_dim  = organ_embs[organ_keys[0]].shape[1]

    # Load whole_ct (may be separate NPZ)
    wct_npz_path = data_cfg.get("whole_ct_embeddings_npz_val")
    if wct_npz_path:
        wct_data = np.load(wct_npz_path, allow_pickle=True)
        wct_ids  = wct_data["sample_ids"].tolist()
        wct_id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(wct_ids)}
        wct_emb  = torch.from_numpy(np.asarray(wct_data[whole_ct_key])).float()
    else:
        wct_id_to_idx = {_normalize_image_id(str(s)): i for i, s in enumerate(sample_ids)}
        wct_emb  = torch.from_numpy(np.asarray(npz[whole_ct_key])).float()

    max_dim    = max(organ_dim, wct_emb.shape[1])
    id_to_idx  = {_normalize_image_id(str(s)): i for i, s in enumerate(sample_ids)}

    # Load val reports
    val_json = data_cfg.get("val_organ_reports_json") or data_cfg.get("val_vqa_json")
    with open(val_json) as f:
        val_reports = json.load(f)
    if max_samples:
        val_reports = val_reports[:max_samples]

    samples = []
    for item in val_reports:
        norm_id = _normalize_image_id(str(item["image"]))
        if norm_id not in id_to_idx:
            continue
        npz_idx = id_to_idx[norm_id]

        emb_list = []
        wct_idx = wct_id_to_idx.get(norm_id)
        wct_vec = wct_emb[wct_idx] if wct_idx is not None else torch.zeros(max_dim)
        if wct_vec.shape[0] < max_dim:
            wct_vec = F.pad(wct_vec, (0, max_dim - wct_vec.shape[0]))
        emb_list.append(wct_vec)

        for k in organ_keys:
            vec = organ_embs[k][npz_idx]
            if vec.shape[0] < max_dim:
                vec = F.pad(vec, (0, max_dim - vec.shape[0]))
            emb_list.append(vec)

        samples.append({
            "image_id":     str(item["image"]),
            "norm_id":      norm_id,
            "embeddings":   torch.stack(emb_list, dim=0),
            "raw_report":   item.get("raw_report", ""),
        })
    return samples



# ── Core generation ────────────────────────────────────────────────────────────

def _apply_repetition_penalty(logits: torch.Tensor, seen_ids: List[int], penalty: float) -> torch.Tensor:
    """Apply repetition penalty (HF RepetitionPenaltyLogitsProcessor equivalent)."""
    if penalty == 1.0 or not seen_ids:
        return logits
    logits = logits.clone()
    score = logits[seen_ids]
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits[seen_ids] = score
    return logits


@torch.no_grad()
def rag_token_generate(
    model,
    tokenizer,
    embeddings: torch.Tensor,
    prompt: str,
    sample_id: str,
    context_index: Optional[dict],
    ret_start_id: int,
    ret_end_id: int,
    rag_token_id: int,
    device: torch.device,
    max_new_tokens: int = 600,
    repetition_penalty: float = 1.2,
    no_rag: bool = False,
    corrupt_rag: bool = False,
    all_context_texts: Optional[List[str]] = None,
    max_retrievals: int = 8,
    cooldown_tokens: int = 30,
    mode: str = "oracle",
    rag_resources: Optional[Dict[str, Any]] = None,
    rag_logit_bias: float = 0.0,
    top_k: int = 3,
    retrieval_stage: str = "two",
) -> Dict[str, Any]:
    """Token-by-token generation with KV cache and [RAG] context injection.

    KV cache invariant:
      - cache is NEVER reset during generation
      - on [RAG] trigger: forward([RAG] + ctx_ids) together in one pass,
        appending to existing cache
      - on normal token: forward(token) in one pass, appending to cache
      - images are processed ONLY in the prefill phase (images=None after that)
    """
    import nltk
    nltk.download('punkt_tab', quiet=True)

    prompt_ids = tokenizer_organ_token(prompt, tokenizer, add_special_tokens=True)
    images_list = project_embeddings(
        embeddings=embeddings.unsqueeze(0), model=model,
        has_images=[True], device=device,
    )

    # ── no-rag fast path: single model.generate call (identical to predict_batch.py) ──
    # COMMENTED OUT to test KV cache loop correctness for --no-rag mode
    # if no_rag:
    #     input_ids_t = tokenizer_organ_token(prompt, tokenizer, return_tensors='pt',
    #                                         add_special_tokens=True).unsqueeze(0).to(device)
    #     attention_mask = torch.ones(1, input_ids_t.shape[1], dtype=torch.long, device=device)
    #     pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    #     with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
    #         out_ids = model.generate(
    #             inputs=input_ids_t,
    #             images=images_list,
    #             attention_mask=attention_mask,
    #             max_new_tokens=max_new_tokens,
    #             do_sample=False,
    #             use_cache=True,
    #             pad_token_id=pad_id,
    #             eos_token_id=tokenizer.eos_token_id,
    #         )
    #     generated_text = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    #     return {
    #         "generated_text":  generated_text,
    #         "retrieval_count": 0,
    #         "retrieval_log":   [],
    #     }

    eos_id = tokenizer.eos_token_id
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    # Verify prompt contains organ tokens (required for image expansion)
    # Organ tokens are stored as negative indices (-201 to -205), NOT tokenizer vocab ids
    from adaragct.constants import ORGAN_TOKEN_INDICES, ORGAN_TOKEN_TO_INDEX
    found_organ = [tid for tid in prompt_ids if tid in ORGAN_TOKEN_INDICES]
    # Also check if organ token strings appear in prompt text (pre-tokenization)
    from adaragct.constants import ORGAN_TOKENS
    organ_in_prompt_text = [t for t in ORGAN_TOKENS if t in prompt]
    # Decode only the positive-id tokens for tail display
    pos_ids = [tid for tid in prompt_ids[-6:] if tid >= 0]

    generated_ids: List[int] = []   # model-predicted tokens only (no ctx)
    all_ids: List[int] = list(prompt_ids)  # full sequence for decoding
    retrieval_log: List[dict] = []
    retrieval_count = 0
    cooldown = 0
    last_organ = "lung"

    # ── Phase 1: Prefill with images via model.generate (max_new_tokens=1) ──
    # MUST use model.generate to trigger LLaVA's prepare_inputs_labels_for_multimodal
    # which replaces negative organ token indices with actual image embeddings.
    # model.forward() directly does NOT trigger this expansion.
    input_ids_t = tokenizer_organ_token(prompt, tokenizer, return_tensors='pt',
                                        add_special_tokens=True).unsqueeze(0).to(device)
    attention_mask = torch.ones(1, input_ids_t.shape[1], dtype=torch.long, device=device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        gen_out = model.generate(
            inputs=input_ids_t,
            images=images_list,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[eos_id, eot_id],
        )

    raw_pkv = gen_out.past_key_values
    first_token_id = int(gen_out.sequences[0, -1].item())

    # model.generate returns a legacy tuple; convert to DynamicCache so that
    # subsequent model.forward() calls (which expect get_seq_length()) work correctly.
    past_key_values = DynamicCache.from_legacy_cache(raw_pkv)

    cache_len = past_key_values.get_seq_length()
    num_organ_tokens = (input_ids_t < 0).sum().item()
    assert cache_len >= input_ids_t.shape[1], (
        f"[ASSERT] cache_len={cache_len} < prompt_tokens={input_ids_t.shape[1]}: KV cache too small"
    )
    assert past_key_values is not None, "[ASSERT] past_key_values is None after prefill"

    # ── Forward first_token to get its KV into cache ──
    # model.generate(max_new_tokens=1) returns past_key_values for the PROMPT only.
    # The first_token was sampled from logits but never forwarded, so its KV is missing.
    # We must forward it now to get correct cache and logits for the next step.
    first_t = torch.tensor([[first_token_id]], dtype=torch.long, device=device)
    total_len = cache_len + 1
    attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        out = model(
            input_ids=first_t,
            images=None,
            attention_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    past_key_values = out.past_key_values
    logits = out.logits[0, -1, :]
    cache_len = past_key_values.get_seq_length()
    generated_ids.append(first_token_id)
    all_ids.append(first_token_id)

    # ── Phase 2: Token-by-token decode ────────────────────────────────
    for step in range(max_new_tokens):
        # Greedy decode with repetition penalty + optional [RAG] logit bias
        logits = _apply_repetition_penalty(logits, all_ids, repetition_penalty)
        if rag_logit_bias != 0.0 and not no_rag:
            logits[rag_token_id] += rag_logit_bias
        token_id = int(logits.argmax(dim=-1).item())

        if token_id in (eos_id, eot_id):
            break

        cooldown = max(0, cooldown - 1)

        # ── Phase 3: [RAG] trigger ────────────────────────────────────
        ctx_ids = None
        if token_id == rag_token_id:
            # [RAG] was generated — determine retrieval strategy
            if no_rag:
                print(f"[RAG] step={step}: [RAG] generated but --no-rag, skipping")
            elif cooldown > 0:
                print(f"[RAG] step={step}: [RAG] generated but cooldown={cooldown}, skipping")
            elif retrieval_count >= max_retrievals:
                print(f"[RAG] step={step}: [RAG] generated but max_retrievals={max_retrievals} reached")

            elif mode in ("normal_rag", "corrupt_rag"):
                # ── FLARE-style probe: generate next sentence, then retrieve ──
                cache_before_rag = cache_len
                print(f"[RAG_TRIGGER] step={step} | mode=normal_rag | "
                      f"cache_before_rag={cache_before_rag}")
                # Forward [RAG] token to get logits for probe
                rag_t = torch.tensor([[token_id]], dtype=torch.long, device=device)
                attn_mask = torch.ones(1, cache_len + 1, dtype=torch.long, device=device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    out = model(
                        input_ids=rag_t, images=None, attention_mask=attn_mask,
                        past_key_values=past_key_values, use_cache=True, return_dict=True,
                    )
                past_key_values = out.past_key_values
                probe_logits = out.logits[0, -1, :]

                # Greedy probe until sentence end (max 60 tokens)
                probe_ids = []
                probe_seen: List[int] = list(all_ids)
                for probe_step in range(60):
                    probe_logits = _apply_repetition_penalty(probe_logits, probe_seen, repetition_penalty)
                    ptid = int(probe_logits.argmax(dim=-1).item())
                    if ptid in (eos_id, eot_id):
                        break
                    probe_ids.append(ptid)
                    probe_seen.append(ptid)
                    probe_decoded = tokenizer.decode(probe_ids, skip_special_tokens=True)
                    if probe_decoded.rstrip().endswith(('.', '?', '!')):
                        break
                    # Forward probe token
                    pt = torch.tensor([[ptid]], dtype=torch.long, device=device)
                    probe_cache_len = past_key_values.get_seq_length()
                    attn_mask = torch.ones(1, probe_cache_len + 1, dtype=torch.long,
                                           device=device)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                        out = model(
                            input_ids=pt, images=None, attention_mask=attn_mask,
                            past_key_values=past_key_values, use_cache=True, return_dict=True,
                        )
                    past_key_values = out.past_key_values
                    probe_logits = out.logits[0, -1, :]

                probe_text = tokenizer.decode(probe_ids, skip_special_tokens=True).strip()
                cache_after_probe = past_key_values.get_seq_length()
                print(f"[PROBE_DONE] step={step} | probe_tokens={len(probe_ids)} | "
                      f"cache_after_probe={cache_after_probe} | "
                      f"probe_text={probe_text!r}")

                # Crop KV cache back to pre-[RAG] state
                past_key_values.crop(cache_before_rag)
                cache_len = past_key_values.get_seq_length()
                assert cache_len == cache_before_rag, (
                    f"[ASSERT] crop failed: expected {cache_before_rag}, got {cache_len}"
                )
                # print(f"[CROP_DONE] cache cropped: {cache_after_probe} → {cache_len}")

                # breakpoint()
                # Skip retrieval if probe_text is too short (e.g. '«.' — no real content)
                probe_content = probe_text.lstrip("«»").strip()
                if len(probe_content) < 5:
                    print(f"[PROBE_SKIP] probe too short ({len(probe_content)} chars), skipping retrieval")
                    # Suppress [RAG] token and set cooldown to prevent death loop
                    token_id = tokenizer.encode(" ", add_special_tokens=False)[0]
                    cooldown = cooldown_tokens
                else:
                    if corrupt_rag and all_context_texts:
                        # ── Corrupt RAG: random sentences instead of real retrieval ──
                        import random
                        ctx_texts = random.sample(all_context_texts, min(top_k, len(all_context_texts)))
                        detected_organ = last_organ
                        print(f"[CORRUPT_RAG] step={step} | injecting {len(ctx_texts)} random sentences")
                    else:
                        # Online retrieval using probe_text
                        gen_text_so_far = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        ctx_texts, detected_organ = online_retrieve_context(
                            probe_text=probe_text,
                            sample_id=sample_id,
                            generated_text=gen_text_so_far,
                            e2e_model=rag_resources["e2e_model"],
                            img2img_top20=rag_resources["img2img_top20"],
                            sentence_data=rag_resources["sentence_data"],
                            sentence_embeds=rag_resources["sentence_embeds"],
                            sid_to_idx=rag_resources["sid_to_idx"],
                            last_organ=last_organ,
                            top_k=top_k,
                            retrieval_stage=retrieval_stage,
                        )
                    last_organ = detected_organ

                    if ctx_texts:
                        ctx_str = f"{RET_START_TOKEN}{' '.join(ctx_texts)}{RET_END_TOKEN}"
                        ctx_ids = tokenizer.encode(ctx_str, add_special_tokens=False)
                        retrieval_log.append({
                            "step": step,
                            "probe_text": probe_text,
                            "organ": detected_organ,
                            "ctx_preview": " ".join(ctx_texts),
                        })
                        # print(f"[RETRIEVE_DONE] step={step} | organ={detected_organ} | "
                            #   f"ctx_tokens={len(ctx_ids)}")
                    else:
                        print(f"[RAG] step={step}: probe done but no context found, "
                              f"organ={detected_organ}")
                        # Suppress [RAG] token and set cooldown
                        token_id = tokenizer.encode(" ", add_special_tokens=False)[0]
                        cooldown = cooldown_tokens

            elif context_index is not None:
                # Oracle mode: precomputed context lookup
                partial_text = tokenizer.decode(generated_ids[-80:], skip_special_tokens=True)
                ctx_texts = lookup_context(
                    context_index, sample_id, partial_text,
                    corrupt=corrupt_rag, all_texts=all_context_texts,
                )
                if ctx_texts:
                    ctx_str = f"{RET_START_TOKEN}{' '.join(ctx_texts)}{RET_END_TOKEN}"
                    ctx_ids = tokenizer.encode(ctx_str, add_special_tokens=False)
                    retrieval_log.append({
                        "step": step,
                        "partial_text": partial_text,
                        "ctx_preview": " ".join(ctx_texts),
                    })
                else:
                    print(f"[RAG] step={step}: [RAG] generated but no context found for "
                          f"sample_id={sample_id!r} partial={repr(partial_text[-40:])}")
                    # Suppress [RAG] token and set cooldown
                    token_id = tokenizer.encode(" ", add_special_tokens=False)[0]
                    cooldown = cooldown_tokens

            else:
                print(f"[RAG] step={step}: [RAG] generated but no retrieval method available")
                token_id = tokenizer.encode(" ", add_special_tokens=False)[0]
                cooldown = cooldown_tokens
        # elif not no_rag and (step < 10 or step % 50 == 0):
        #     # Periodic health check: log current generation state + [RAG] prob
        #     logits_f = logits.float()
        #     probs = torch.softmax(logits_f, dim=-1)
        #     max_prob, max_idx = probs.max(dim=-1)
        #     entropy = -(probs * (probs + 1e-10).log()).sum().item()
        #     gen_so_far = tokenizer.decode(generated_ids[-20:], skip_special_tokens=True)
        #     rag_prob = probs[rag_token_id].item()
        #     # Top-5 tokens at this position
        #     top5_probs, top5_ids = probs.topk(5)
        #     top5_str = " | ".join(
        #         f"{repr(tokenizer.decode([tid.item()]))}: {p.item():.4f}"
        #         for tid, p in zip(top5_ids, top5_probs)
        #     )
        #     print(f"[HEALTH] step={step} cache={cache_len} generated={len(generated_ids)} "
        #           f"entropy={entropy:.2f} max_prob={max_prob.item():.3f} "
        #           f"next_token={repr(tokenizer.decode([max_idx.item()]))} "
        #           f"gen_tail={repr(gen_so_far[-30:])}")
        #     print(f"  [RAG_PROB] p([RAG])={rag_prob:.6f} | top5: {top5_str}")
        if token_id != rag_token_id and ctx_ids is None:
            pass  # normal path, no extra logic needed

        # ── Phase 4: Forward pass (with or without context) ───────────
        if ctx_ids is not None:
            # Forward [RAG] token + ctx_ids together in one pass
            # This is the ONLY forward for [RAG]; ctx tokens are also forwarded once here.
            inject_ids = [token_id] + ctx_ids
            inject_t   = torch.tensor([inject_ids], dtype=torch.long, device=device)
            cache_before = cache_len
            total_len  = cache_len + len(inject_ids)
            attn_mask  = torch.ones(1, total_len, dtype=torch.long, device=device)

            print(f"[INJECT] step={step} retrieval={retrieval_count+1} "
                  f"cache_before={cache_before} inject_len={len(inject_ids)} "
                  f"(1 [RAG] + {len(ctx_ids)} ctx) attn_mask_len={total_len}")

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out = model(
                    input_ids=inject_t,
                    images=None,          # CRITICAL: skip image re-processing
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = out.past_key_values
            logits          = out.logits[0, -1, :]
            cache_len       = past_key_values.get_seq_length()

            # Verify: cache grew by exactly len(inject_ids)
            expected_cache = cache_before + len(inject_ids)
            assert cache_len == expected_cache, (
                f"[ASSERT] inject cache mismatch: expected {expected_cache} "
                f"(before={cache_before} + inject={len(inject_ids)}), got {cache_len}"
            )

            # [RAG] → generated_ids; ctx → all_ids only (not generated)
            generated_ids.append(token_id)
            all_ids.append(token_id)
            all_ids.extend(ctx_ids)
            retrieval_count += 1
            cooldown = cooldown_tokens

        else:
            # Normal single-token forward
            next_t    = torch.tensor([[token_id]], dtype=torch.long, device=device)
            cache_before = cache_len
            total_len = cache_len + 1
            attn_mask = torch.ones(1, total_len, dtype=torch.long, device=device)

            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                out = model(
                    input_ids=next_t,
                    images=None,
                    attention_mask=attn_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = out.past_key_values
            logits          = out.logits[0, -1, :]
            cache_len       = past_key_values.get_seq_length()

            # Verify: cache grew by exactly 1
            assert cache_len == cache_before + 1, (
                f"[ASSERT] normal forward cache mismatch at step={step}: "
                f"expected {cache_before + 1}, got {cache_len}"
            )

            generated_ids.append(token_id)
            all_ids.append(token_id)


    # ── Final cache audit ──
    total_ctx_tokens = len(all_ids) - len(prompt_ids) - len(generated_ids)
    img_expansion = cache_len - len(all_ids)
    print(f"\n{'='*60}")
    print(f"[CACHE AUDIT] Final KV cache state:")
    print(f"  cache_len          = {cache_len}")
    print(f"  prompt_tokens      = {len(prompt_ids)}")
    print(f"  image_expansion    = {img_expansion} (organ tokens → embeddings)")
    print(f"  generated_tokens   = {len(generated_ids)} (model output, incl [RAG])")
    print(f"  ctx_injected_tokens= {total_ctx_tokens} (from {retrieval_count} retrievals)")
    print(f"  all_ids            = {len(all_ids)} (prompt + generated + ctx)")
    print(f"  expected_cache     = {len(all_ids)} + {img_expansion} = {len(all_ids) + img_expansion}")
    print(f"  actual_cache       = {cache_len}")
    print(f"  MATCH: {cache_len == len(all_ids) + img_expansion}")
    print(f"{'='*60}")
    assert cache_len == len(all_ids) + img_expansion, (
        f"[ASSERT] Final cache audit FAILED: cache_len={cache_len} != "
        f"all_ids({len(all_ids)}) + img_expansion({img_expansion}) = {len(all_ids) + img_expansion}"
    )

    # Strip any residual <|ret_start|>...<|ret_end|> from generated_ids
    # (shouldn't happen since ctx_ids never go into generated_ids, but safety check)
    clean_ids, in_ctx = [], False
    for tid in generated_ids:
        if tid == ret_start_id:
            in_ctx = True
        elif tid == ret_end_id:
            in_ctx = False
        elif not in_ctx:
            clean_ids.append(tid)

    generated_text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()

    return {
        "generated_text":  generated_text,
        "retrieval_count": retrieval_count,
        "retrieval_log":   retrieval_log,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Token Inference")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint dir (e.g. .../checkpoints/step_3000)")
    parser.add_argument("--config", default=None,
                        help="Path to training YAML config (optional, auto-loaded from checkpoint training_state.pt)")
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle RAG: use precomputed oracle context (hardcoded path)")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable all RAG injection (baseline ablation)")
    parser.add_argument("--corrupt-rag", action="store_true",
                        help="Inject random wrong context (ablation study, TODO)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--max-retrievals", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of context sentences to retrieve per [RAG] trigger (default: 3)")
    parser.add_argument("--rag-logit-bias", type=float, default=0.0,
                        help="Positive bias added to [RAG] token logit to increase trigger frequency (default: 0.0)")
    parser.add_argument("--cooldown-tokens", type=int, default=5,
                        help="Min tokens between [RAG] triggers (default: 5)")
    parser.add_argument("--output-suffix", type=str, default=None,
                        help="Suffix appended to output subdir name for experiment distinction (e.g. 'bias3')")
    parser.add_argument("--global-context", action="store_true",
                        help="Hierarchical RAG: prepend whole_ct top-K similar reports as global context, then do adaptive organ RAG")
    parser.add_argument("--global-context-topk", type=int, default=3,
                        help="Number of similar images for global context (default: 3)")
    parser.add_argument("--text2text", action="store_true",
                        help="One-stage retrieval: direct text-to-text over full database "
                             "(skip img2img Stage 1). Auto-suffix: infer_normal_rag_text2text")
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (auto-inferred if omitted)")
    args = parser.parse_args()

    # Load config: prefer training_state.pt inside checkpoint (has full config),
    # fallback to --config yaml if provided
    ckpt_path = Path(args.checkpoint)
    training_state_path = ckpt_path / "training_state.pt"
    if training_state_path.exists():
        ts = torch.load(training_state_path, map_location="cpu", weights_only=False)
        config = ts["config"]
        print(f"Config loaded from training_state.pt (step={ts.get('step', '?')})")
    elif args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        print(f"Config loaded from {args.config}")
    else:
        raise FileNotFoundError(
            f"No training_state.pt in {ckpt_path} and no --config provided."
        )

    # Determine mode
    ORACLE_CONTEXT_PATH = "data/oracle/oracle_context_top3.jsonl"
    if args.no_rag:
        mode = "no_rag"
    elif args.oracle:
        mode = "oracle"
    elif args.corrupt_rag:
        mode = "corrupt_rag"
    else:
        mode = "normal_rag"

    # --text2text implies normal_rag + one-stage retrieval
    args.retrieval_stage = "one" if args.text2text else "two"
    if args.text2text:
        if args.no_rag:
            print("[WARN] --text2text ignored in --no-rag mode")
            args.text2text = False
            args.retrieval_stage = "two"
        else:
            mode = "normal_rag"  # force normal_rag mode for text2text
    print(f"Mode: {mode}" + (" (text2text one-stage)" if args.text2text else ""))

    if args.no_rag and args.rag_logit_bias != 0.0:
        print(f"[WARN] --rag-logit-bias={args.rag_logit_bias} ignored in --no-rag mode")
    if args.rag_logit_bias != 0.0:
        print(f"[RAG] logit bias: {args.rag_logit_bias}, cooldown: {args.cooldown_tokens}")

    if args.global_context:
        print(f"[GLOBAL] Hierarchical RAG: global context (top-{args.global_context_topk}) + adaptive organ RAG")

    # Auto-infer output path
    if args.output is None:
        ckpt_path = Path(args.checkpoint)
        step_name = ckpt_path.name
        subdir = f"infer_{mode}"
        if args.text2text:
            subdir = f"{subdir}_text2text"
        if args.global_context:
            subdir = f"{subdir}_gc{args.global_context_topk}"
        if args.output_suffix:
            subdir = f"{subdir}_{args.output_suffix}"
        args.output = str(ckpt_path.parent.parent / subdir / f"{step_name}.jsonl")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer, ret_start_id, ret_end_id, rag_token_id = load_model(args.checkpoint, device)

    # Load context index based on mode
    context_index = None
    all_context_texts = None
    rag_resources = None

    if mode == "no_rag":
        print("Mode: no-RAG (baseline) — context not loaded")

    elif mode == "oracle":
        assert os.path.exists(ORACLE_CONTEXT_PATH), \
            f"[ASSERT] Oracle context not found: {ORACLE_CONTEXT_PATH}"
        context_index = build_context_index(ORACLE_CONTEXT_PATH)
        print(f"Mode: oracle RAG — context index loaded ({len(context_index)} entries)")

    elif mode == "normal_rag":
        # Load online retrieval resources
        e2e_model = load_text_encoder(device)
        sentence_data, sentence_embeds, sid_to_idx, sid_to_sent = load_sentence_db()
        img2img_top20 = load_img2img_top20()
        rag_resources = {
            "e2e_model": e2e_model,
            "img2img_top20": img2img_top20,
            "sentence_data": sentence_data,
            "sentence_embeds": sentence_embeds,
            "sid_to_idx": sid_to_idx,
        }
        print(f"Mode: normal_rag — online retrieval resources loaded")

    # ── Global context (hierarchical RAG) ──
    global_context_map = None
    if args.global_context:
        print(f"[GLOBAL] Loading whole_ct img2img for global context (top-{args.global_context_topk})...")
        with open(IMG2IMG_WHOLE_CT_PATH) as f:
            img2img_whole_ct = json.load(f)
        print(f"  {len(img2img_whole_ct)} samples")
        with open(SENTENCE_DB_PATH) as f:
            gc_sentence_db = json.load(f)
        # Build report index: image_id -> full report text
        report_index = {img_id: " ".join(s["text"] for s in sents)
                        for img_id, sents in gc_sentence_db.items()}
        print(f"  Report index: {len(report_index)} images")
        # Precompute global context per validation sample
        global_context_map = {}
        for sid, entries in img2img_whole_ct.items():
            reports = []
            for entry in entries[:args.global_context_topk]:
                r = report_index.get(entry["image_id"], "")
                if r:
                    reports.append(r)
            if reports:
                global_context_map[sid] = " ".join(reports)
        print(f"  Global context built for {len(global_context_map)} samples")

    elif mode == "corrupt_rag":
        # Load sentence DB for random sampling pool
        sentence_data, sentence_embeds, sid_to_idx, sid_to_sent = load_sentence_db()
        # Collect all sentence texts into a flat list
        all_context_texts = []
        for img_id, sents in sentence_data.items():
            for s in sents:
                all_context_texts.append(s["text"])
        print(f"Mode: corrupt_rag — random pool: {len(all_context_texts)} sentences")
        # rag_resources not needed for corrupt (no real retrieval), but needed for probe flow
        rag_resources = {
            "e2e_model": None,
            "img2img_top20": None,
            "sentence_data": sentence_data,
            "sentence_embeds": sentence_embeds,
            "sid_to_idx": sid_to_idx,
        }

    print("Loading validation data...")
    samples = load_validation_data(config, max_samples=args.max_samples)
    print(f"  {len(samples)} samples loaded")

    # Load already-completed predictions for incremental inference
    done_ids = set()
    if output_path.exists():
        with open(output_path) as _f:
            for _line in _f:
                try:
                    _r = json.loads(_line)
                    done_ids.add(_r["image"])
                except Exception:
                    pass
        print(f"  Resuming: {len(done_ids)} samples already done, skipping them.")
        samples = [s for s in samples if s["image_id"] not in done_ids]
    print(f"  {len(samples)} samples pending")

    # Build prompt (assistant header included, no target)
    system_prompt = SYSTEM_PROMPTS[config["data"]["system_prompt"]]
    user_content = (
        f"{ORGAN_TOKENS_DESC}\n{TASK_TOKENS['report_generation']}\n"
        "Would you mind generating the radiology report for the specified chest CT scan?"
    )
    (prompt,) = _build_chat_prompt(system_prompt, user_content)

    results = []
    total_retrievals = 0

    out_f = open(output_path, "a")

    for si, sample in enumerate(tqdm(samples, desc="Inference")):
        # Prepend global context if enabled
        sample_prompt = prompt
        gc_words = 0
        if global_context_map is not None:
            gc_text = global_context_map.get(sample["norm_id"])
            if gc_text:
                gc_words = len(gc_text.split())
                sample_prompt = f"{RET_START_TOKEN} Reference context: {gc_text} {RET_END_TOKEN}\n" + prompt

        out = rag_token_generate(
            model=model,
            tokenizer=tokenizer,
            embeddings=sample["embeddings"].to(device),
            prompt=sample_prompt,
            sample_id=sample["norm_id"],
            context_index=context_index,
            ret_start_id=ret_start_id,
            ret_end_id=ret_end_id,
            rag_token_id=rag_token_id,
            device=device,
            max_new_tokens=args.max_new_tokens,
            no_rag=(mode == "no_rag"),
            corrupt_rag=(mode == "corrupt_rag"),
            all_context_texts=all_context_texts,
            max_retrievals=args.max_retrievals,
            cooldown_tokens=args.cooldown_tokens,
            mode=mode,
            rag_resources=rag_resources,
            rag_logit_bias=args.rag_logit_bias,
            top_k=args.top_k,
            retrieval_stage=args.retrieval_stage,
        )

        total_retrievals += out["retrieval_count"]
        result = {
            "image":           sample["image_id"],
            "generated_text":  out["generated_text"],
            "reference_text":  sample["raw_report"],
            "task_type":       "report_generation",
            "retrieval_count": out["retrieval_count"],
            "retrieval_log":   out["retrieval_log"],
            "rag_logit_bias":  args.rag_logit_bias,
            "cooldown_tokens": args.cooldown_tokens,
            "global_context_words": gc_words,
        }
        results.append(result)
        out_f.write(json.dumps(result) + "\n")
        out_f.flush()

        sep = "-" * 60
        tqdm.write(f"\n{sep}")
        gc_info = f" | gc_words={gc_words}" if global_context_map is not None else ""
        tqdm.write(f"[{si+1}/{len(samples)}] image={sample['image_id']} | retrievals={out['retrieval_count']}{gc_info}")
        if global_context_map is not None and gc_words > 0:
            tqdm.write(f"[CTX]  {gc_text[:500]}")
        tqdm.write(f"[GEN]  {out['generated_text']}")
        tqdm.write(f"[REF]  {sample['raw_report']}")
        tqdm.write(sep)

    out_f.close()

    avg_ret = total_retrievals / len(results) if results else 0
    print(f"\nSaved {len(results)} new predictions → {output_path}")
    print(f"Avg retrievals per sample: {avg_ret:.2f}")


if __name__ == "__main__":
    main()
