"""
Step 1.3: Pre-compute oracle contexts for training.

For each needs_rag=True sentence:
1. Get top-20 similar training samples (from Step 1.2, same organ)
2. Collect all sentences from those samples for the same organ
3. Compute Txt2Txt cosine similarity between GT sentence embedding and candidates
4. Pick the best-matching sentence as oracle context

Depends on:
- Step 1.1: sentence_perplexity.json (which sentences need RAG)
- Step 1.2: stage1_train_top20_{organ}.json (which samples to search)
- Database: train_{organ}_sentences.json + train_{organ}_sentence_embeds.npz

CPU only (no model forward, just numpy dot products).

Usage:
    python P2_rag/oracle/precompute_oracle_contexts.py
    python P2_rag/oracle/precompute_oracle_contexts.py --top-k-sentences 3
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from config import ORGAN_NAMES

ORACLE_DIR = "results/P2_rag/oracle"
DATABASE_DIR = "results/P2_rag/database"


def normalize_id(name: str) -> str:
    """Normalize volume name."""
    s = str(name).strip()
    s = Path(s).name
    if s.endswith('.nii.gz'):
        s = s[:-7]
    elif s.endswith('.nii'):
        s = s[:-4]
    return s


def load_sentence_database(organ: str, split: str = "train"):
    """Load sentence texts and embeddings for an organ.

    Returns:
        sentences: list of {sample_id, sentence_id, text}
        embeddings: (N, 256) numpy array
        sample_to_indices: {sample_id: [idx1, idx2, ...]}
    """
    json_path = Path(DATABASE_DIR) / f"{split}_{organ}_sentences.json"
    npz_path = Path(DATABASE_DIR) / f"{split}_{organ}_sentence_embeds.npz"

    assert json_path.exists(), f"Missing: {json_path}"
    assert npz_path.exists(), f"Missing: {npz_path}"

    with open(json_path) as f:
        sentences = json.load(f)

    npz = np.load(npz_path, allow_pickle=True)
    embeddings = npz["embeddings"].astype(np.float32)

    # Build sample_id -> sentence indices mapping
    sample_to_indices = defaultdict(list)
    for i, sent in enumerate(sentences):
        sid = normalize_id(sent["sample_id"])
        sample_to_indices[sid].append(i)

    return sentences, embeddings, sample_to_indices


def find_best_context(
    gt_embedding: np.ndarray,
    candidate_indices: list,
    all_embeddings: np.ndarray,
    all_sentences: list,
    exclude_sample_id: str,
) -> dict:
    """Find the best matching sentence from candidates via Txt2Txt cosine similarity.

    Args:
        gt_embedding: (256,) embedding of the GT sentence
        candidate_indices: list of indices into all_embeddings/all_sentences
        all_embeddings: (N, 256) all sentence embeddings
        all_sentences: list of sentence dicts
        exclude_sample_id: sample to exclude (self)

    Returns:
        dict with retrieved_sentence, retrieved_from, similarity
    """
    assert candidate_indices, "candidate_indices is empty"

    # Filter out self
    filtered = [
        idx for idx in candidate_indices
        if normalize_id(all_sentences[idx]["sample_id"]) != exclude_sample_id
    ]
    assert filtered, f"All candidates are self ({exclude_sample_id}), no valid neighbors"

    # Compute cosine similarity
    candidate_embs = all_embeddings[filtered]  # (K, 256)
    gt_norm = gt_embedding / (np.linalg.norm(gt_embedding) + 1e-8)
    cand_norms = candidate_embs / (np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8)
    similarities = cand_norms @ gt_norm  # (K,)

    best_idx_in_filtered = np.argmax(similarities)
    best_global_idx = filtered[best_idx_in_filtered]
    best_sim = float(similarities[best_idx_in_filtered])

    return {
        "retrieved_sentence": all_sentences[best_global_idx]["text"],
        "retrieved_from": normalize_id(all_sentences[best_global_idx]["sample_id"]),
        "similarity": round(best_sim, 4),
        "sentence_id": all_sentences[best_global_idx]["sentence_id"],
    }


def main():
    parser = argparse.ArgumentParser(description="Step 1.3: Pre-compute oracle contexts")
    parser.add_argument("--top-k-sentences", type=int, default=1,
                        help="Number of oracle context sentences per needs_rag sentence")
    parser.add_argument("--oracle-dir", type=str, default=ORACLE_DIR)
    args = parser.parse_args()

    oracle_dir = Path(args.oracle_dir)

    # Load Step 1.1 output
    ppl_path = oracle_dir / "sentence_perplexity.json"
    assert ppl_path.exists(), f"{ppl_path} not found. Run precompute_perplexity.py first."
    print(f"Loading perplexity data: {ppl_path}")
    with open(ppl_path) as f:
        ppl_data = json.load(f)
    print(f"  {len(ppl_data)} samples")

    # Count needs_rag sentences per organ
    needs_rag_count = defaultdict(int)
    total_count = defaultdict(int)
    for sample_id, organs in ppl_data.items():
        for organ, sentences in organs.items():
            for sent in sentences:
                total_count[organ] += 1
                if sent["needs_rag"]:
                    needs_rag_count[organ] += 1

    print(f"\n  needs_rag counts:")
    for organ in ORGAN_NAMES + ["other"]:
        n_rag = needs_rag_count[organ]
        n_total = total_count[organ]
        pct = n_rag / n_total * 100 if n_total > 0 else 0
        print(f"    {organ:12s}: {n_rag:6d}/{n_total:6d} ({pct:.1f})")

    # Load sentence databases per organ
    print(f"\nLoading sentence databases...")
    organ_db = {}
    for organ in ORGAN_NAMES:
        sentences, embeddings, sample_to_indices = load_sentence_database(organ)
        organ_db[organ] = {
            "sentences": sentences,
            "embeddings": embeddings,
            "sample_to_indices": sample_to_indices,
        }
        print(f"  {organ}: {len(sentences)} sentences, embeddings {embeddings.shape}")

    # Load Step 1.2 outputs (train→train top-K per organ)
    print(f"\nLoading train→train top-K mappings...")
    organ_topk = {}
    for organ in ORGAN_NAMES:
        topk_path = oracle_dir / f"stage1_train_top20_{organ}.json"
        assert topk_path.exists(), f"Missing: {topk_path}. Run precompute_train_topk.py first."
        with open(topk_path) as f:
            organ_topk[organ] = json.load(f)
        print(f"  {organ}: {len(organ_topk[organ])} queries loaded")

    # Process each sample
    print(f"\n{'='*70}")
    print(f"Computing oracle contexts")
    print(f"{'='*70}")

    oracle_contexts = {}
    stats = defaultdict(int)

    for sample_id in tqdm(ppl_data, desc="Oracle matching"):
        sample_contexts = {}

        for organ in ORGAN_NAMES:
            sentences = ppl_data[sample_id][organ]
            organ_contexts = []

            for sent in sentences:
                if not sent["needs_rag"]:
                    continue

                # Get top-K similar samples for this organ
                topk_samples = organ_topk[organ][sample_id]

                # Collect candidate sentence indices from top-K samples
                db = organ_db[organ]
                candidate_indices = []
                for neighbor_id in topk_samples:
                    candidate_indices.extend(db["sample_to_indices"][neighbor_id])

                if not candidate_indices:
                    continue

                # Find GT sentence embedding
                gt_indices = db["sample_to_indices"][sample_id]
                gt_embedding = None
                for gi in gt_indices:
                    if db["sentences"][gi]["text"] == sent["text"]:
                        gt_embedding = db["embeddings"][gi]
                        break
                if gt_embedding is None:
                    for gi in gt_indices:
                        if sent["text"][:30] in db["sentences"][gi]["text"]:
                            gt_embedding = db["embeddings"][gi]
                            break
                assert gt_embedding is not None, \
                    f"GT embedding not found for sentence: {sent['text'][:60]!r} in {sample_id}/{organ}"

                # Find best context
                result = find_best_context(
                    gt_embedding=gt_embedding,
                    candidate_indices=candidate_indices,
                    all_embeddings=db["embeddings"],
                    all_sentences=db["sentences"],
                    exclude_sample_id=sample_id,
                )
                organ_contexts.append({
                    "sentence_idx": sent["sentence_idx"],
                    "gt_sentence": sent["text"],
                    "needs_rag": True,
                    **result,
                })
                stats["matched"] += 1

            if organ_contexts:
                sample_contexts[organ] = organ_contexts

        if sample_contexts:
            oracle_contexts[sample_id] = sample_contexts

    # Print stats
    print(f"\n{'='*70}")
    print(f"ORACLE CONTEXT MATCHING RESULTS")
    print(f"{'='*70}")
    print(f"  Samples with contexts: {len(oracle_contexts)}/{len(ppl_data)}")
    print(f"  Successfully matched: {stats['matched']}")
    print(f"  No top-K neighbors: {stats['no_topk']}")
    print(f"  No database: {stats['no_db']}")
    print(f"  No candidates: {stats['no_candidates']}")
    print(f"  No GT embedding: {stats['no_gt_embedding']}")
    print(f"  No match found: {stats['no_match']}")

    # Per-organ stats
    print(f"\n  Per-organ matched contexts:")
    for organ in ORGAN_NAMES:
        n = sum(
            len(oracle_contexts.get(sid, {}).get(organ, []))
            for sid in oracle_contexts
        )
        print(f"    {organ:12s}: {n:6d} contexts")

    # Similarity distribution
    all_sims = []
    for sid, organs in oracle_contexts.items():
        for organ, contexts in organs.items():
            for ctx in contexts:
                all_sims.append(ctx["similarity"])
    if all_sims:
        print(f"\n  Similarity distribution:")
        print(f"    Mean: {np.mean(all_sims):.4f}")
        print(f"    Median: {np.median(all_sims):.4f}")
        print(f"    P25: {np.percentile(all_sims, 25):.4f}")
        print(f"    P75: {np.percentile(all_sims, 75):.4f}")

    # Save
    output_path = oracle_dir / "oracle_contexts.json"
    with open(output_path, "w") as f:
        json.dump(oracle_contexts, f, indent=2)
    print(f"\n  Saved: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
