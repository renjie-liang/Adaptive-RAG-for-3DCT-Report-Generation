"""
Step 4: Evaluation pipeline for oracle-trained models.

Computes NLP metrics (BLEU, ROUGE, BERTScore) and clinical F1
on predictions from inference.py.

Usage:
    python P2_rag/oracle/evaluate.py \
        --predictions results/P2_rag/oracle/oracle_raw_entropy/predictions.json \
        --output results/P2_rag/oracle/oracle_raw_entropy/metrics.json

    # Compare multiple experiments:
    python P2_rag/oracle/evaluate.py \
        --predictions results/P2_rag/oracle/*/predictions.json \
        --output results/P2_rag/oracle/comparison.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def compute_bleu(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU-1/2/4 scores."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt_tab', quiet=True)
        from nltk.tokenize import word_tokenize

        smooth = SmoothingFunction().method1
        hyp_tokens = [word_tokenize(h.lower()) for h in hypotheses]
        ref_tokens = [[word_tokenize(r.lower())] for r in references]

        return {
            "bleu1": corpus_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth),
            "bleu2": corpus_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
            "bleu4": corpus_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth),
        }
    except Exception as e:
        print(f"  BLEU failed: {e}")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}


def compute_rouge(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE-1/2/L scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = defaultdict(list)
        for hyp, ref in zip(hypotheses, references):
            s = scorer.score(ref, hyp)
            for k, v in s.items():
                scores[k].append(v.fmeasure)
        return {k: float(np.mean(v)) for k, v in scores.items()}
    except Exception as e:
        print(f"  ROUGE failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bertscore(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BERTScore F1."""
    try:
        from bert_score import score as bert_score
        P, R, F = bert_score(hypotheses, references, lang="en", verbose=False)
        return {
            "bertscore_p": float(P.mean()),
            "bertscore_r": float(R.mean()),
            "bertscore_f": float(F.mean()),
        }
    except Exception as e:
        print(f"  BERTScore failed: {e}")
        return {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f": 0.0}


DEFAULT_CLINICAL_LABELS_CSV = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"
DEFAULT_RADBERT_CHECKPOINT = "/orange/xujie/liang.renjie/DATA/weights/CT-CLIP/RadBertClassifier.pth"


def compute_clinical_f1(
    predictions: List[Dict],
    hypotheses: List[str],
    radbert_checkpoint: str = DEFAULT_RADBERT_CHECKPOINT,
    clinical_labels_csv: str = DEFAULT_CLINICAL_LABELS_CSV,
) -> Dict[str, float]:
    """Compute clinical F1 using RadBERT classifier + GT labels from CSV."""
    try:
        from adaragct.eval.clinical_efficacy import (
            extract_findings, compute_clinical_f1 as _compute_f1,
            load_clinical_labels_csv, _normalize_volume_name,
            CLINICAL_FINDINGS,
        )

        # Load GT labels from CSV (same source as stage3_turn/cal_metrics.py)
        labels_by_vol = load_clinical_labels_csv(clinical_labels_csv)

        # Align predictions with GT labels
        y_true_list = []
        valid_hyps = []
        missing = 0
        for pred, hyp in zip(predictions, hypotheses):
            key = _normalize_volume_name(pred.get("image", ""))
            if key not in labels_by_vol:
                missing += 1
                continue
            y_true_list.append(labels_by_vol[key])
            valid_hyps.append(hyp)

        if missing > 0:
            print(f"  Clinical: {missing} samples missing in CSV, using {len(valid_hyps)} aligned")

        if not valid_hyps:
            raise ValueError("No samples could be aligned with clinical labels CSV")

        y_true = np.array(y_true_list, dtype=int)
        y_pred = extract_findings(valid_hyps, checkpoint_path=radbert_checkpoint)

        results = _compute_f1(y_true, y_pred, CLINICAL_FINDINGS)
        return {
            "clinical_precision": results["precision"],
            "clinical_recall": results["recall"],
            "clinical_f1": results["f1"],
            "clinical_micro_f1": results["micro_f1"],
            "clinical_macro_f1": results["macro_f1"],
        }
    except Exception as e:
        print(f"  Clinical F1 failed: {e}")
        return {
            "clinical_precision": 0.0,
            "clinical_recall": 0.0,
            "clinical_f1": 0.0,
            "clinical_micro_f1": 0.0,
            "clinical_macro_f1": 0.0,
        }


def evaluate_predictions(predictions_path: str) -> Dict:
    """Evaluate a single predictions file (JSONL format)."""
    with open(predictions_path) as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    hypotheses = [p["generated_text"] for p in predictions]
    references = [p["reference_text"] for p in predictions]

    # Filter empty (keep aligned predictions list for clinical label lookup)
    valid_indices = [i for i, (h, r) in enumerate(zip(hypotheses, references)) if h.strip() and r.strip()]
    if not valid_indices:
        print(f"  WARNING: No valid predictions in {predictions_path}")
        return {}

    valid_preds = [predictions[i] for i in valid_indices]
    hypotheses = [hypotheses[i] for i in valid_indices]
    references = [references[i] for i in valid_indices]

    print(f"  Evaluating {len(hypotheses)} samples...")

    metrics = {}
    metrics.update(compute_bleu(hypotheses, references))
    metrics.update(compute_rouge(hypotheses, references))
    metrics.update(compute_bertscore(hypotheses, references))
    metrics.update(compute_clinical_f1(valid_preds, hypotheses))

    # Retrieval stats
    retrieval_counts = [p.get("retrieval_count", 0) for p in predictions]
    metrics["avg_retrievals"] = float(np.mean(retrieval_counts))
    metrics["pct_with_retrieval"] = float(np.mean([c > 0 for c in retrieval_counts]))
    metrics["n_samples"] = len(predictions)
    metrics["n_valid"] = len(hypotheses)

    return metrics


def print_metrics(metrics: Dict, name: str = ""):
    """Pretty-print metrics."""
    header = f"  {name}" if name else "  Results"
    print(f"\n{header}")
    print(f"  {'─'*60}")
    groups = [
        ("NLP", ["bleu1", "bleu2", "bleu4", "rouge1", "rouge2", "rougeL"]),
        ("BERTScore", ["bertscore_p", "bertscore_r", "bertscore_f"]),
        ("Clinical", ["clinical_precision", "clinical_recall", "clinical_f1", "clinical_micro_f1", "clinical_macro_f1"]),
        ("Retrieval", ["avg_retrievals", "pct_with_retrieval"]),
    ]
    for group_name, keys in groups:
        vals = {k: metrics[k] for k in keys if k in metrics}
        if vals:
            print(f"  {group_name}:")
            for k, v in vals.items():
                print(f"    {k:25s}: {v:.4f}")


def _infer_metrics_dir(pred_path: Path) -> Path:
    """Auto-derive metrics output dir from infer subdir name.

    infer/           -> metrics/
    infer_no_rag/    -> metrics_no_rag/
    infer_oracle_rag/ -> metrics_oracle_rag/
    """
    infer_subdir = pred_path.parent.name  # e.g. "infer_no_rag"
    metrics_subdir = infer_subdir.replace("infer", "metrics", 1)
    return pred_path.parent.parent / metrics_subdir


def main():
    parser = argparse.ArgumentParser(description="Evaluate oracle model predictions")
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="Path(s) to .jsonl prediction files (supports glob)")
    args = parser.parse_args()

    # Expand globs
    prediction_files = []
    for p in args.predictions:
        matches = list(Path(".").glob(p)) if "*" in p else [Path(p)]
        prediction_files.extend(matches)
    prediction_files = [f for f in prediction_files if f.exists()]

    if not prediction_files:
        print("ERROR: No prediction files found.")
        sys.exit(1)

    print(f"Evaluating {len(prediction_files)} prediction file(s)...")

    all_results = {}
    for pred_path in sorted(prediction_files):
        pred_path = Path(pred_path)
        exp_name = pred_path.parent.parent.name  # R01_xxx
        infer_tag = pred_path.parent.name         # infer / infer_no_rag / infer_oracle_rag
        label = f"{exp_name}/{infer_tag}"

        print(f"\n{'='*70}")
        print(f"Experiment: {label}")
        print(f"File: {pred_path}")
        print(f"{'='*70}")

        metrics = evaluate_predictions(str(pred_path))
        all_results[label] = {"path": str(pred_path), "metrics": metrics}
        print_metrics(metrics, label)

        # Auto-save metrics JSON next to predictions in metrics_*/ dir
        metrics_dir = _infer_metrics_dir(pred_path)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_out = metrics_dir / pred_path.name.replace("infer_", "metrics_")
        with open(metrics_out, "w") as f:
            json.dump({"label": label, "predictions": str(pred_path), "metrics": metrics}, f, indent=2)
        print(f"  Saved metrics → {metrics_out}")

    # Summary table if multiple experiments
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("COMPARISON TABLE")
        print(f"{'='*70}")
        key_metrics = ["bleu4", "rouge1", "rougeL", "bertscore_f",
                       "clinical_f1", "clinical_macro_f1", "avg_retrievals"]
        header = f"{'Experiment':50s}" + "".join(f"{k:18s}" for k in key_metrics)
        print(header)
        print("─" * len(header))
        for label, result in sorted(all_results.items()):
            m = result["metrics"]
            row = f"{label:50s}" + "".join(
                f"{m.get(k, 0.0):18.4f}" for k in key_metrics
            )
            print(row)


if __name__ == "__main__":
    main()
