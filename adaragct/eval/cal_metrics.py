"""
Stage 3 Turn Metrics Calculation

Calculate evaluation metrics from prediction JSONL files.

Metrics computed:
- All tasks: BLEU (1-4), METEOR, ROUGE-L, CIDEr, Llama Score
- Report generation additionally: Precision, Recall, F1 (clinical efficacy)

Usage:
    python cal_metrics.py --predictions predictions.jsonl --output metrics.json
"""
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional
from collections import defaultdict, OrderedDict
import csv
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaragct.eval.text_metrics import compute_all_text_metrics
from adaragct.eval.llama_score import compute_llama_score
# GREEN score removed — install separately if needed
# from adaragct.eval.green_score import GREEN
from adaragct.eval.clinical_efficacy import (
    CLINICAL_FINDINGS,
    compute_clinical_efficacy_from_labels,
    extract_findings,
    compute_clinical_f1,
)
from adaragct.utils.path_utils import generate_metrics_path_from_predictions


# Task order for consistent output
TASK_ORDER = ["report_generation", "long_answer", "short_answer", "multiple_choice"]

_HYP_KEYS = ["generated_text", "prediction", "full_generated"]
_REF_KEYS = ["reference_text", "reference", "full_reference"]
_Q_KEYS   = ["question", "input_prompt"]

# Unified prompt for report_generation task (used as question in metrics)
REPORT_GENERATION_PROMPT = "Would you mind generating the radiology report for the specified chest CT scan?"


def _extract(pred: Dict, keys: list) -> str:
    """Return the first matching key's value, or raise KeyError."""
    for k in keys:
        if k in pred:
            return pred[k]
    raise KeyError(f"None of {keys} found in prediction with keys {list(pred.keys())}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Calculate metrics from predictions")

    parser.add_argument('predictions', type=str, help='Path to predictions JSONL file')
    parser.add_argument('--output', type=str, default=None, help='Path to save metrics JSON')
    parser.add_argument('--compute-green', action=argparse.BooleanOptionalAction, default=False, help='Compute GREEN score (LLM-based). Use --compute-green to enable.')
    parser.add_argument('--green-model-path', type=str,
                        default="/path/to/GREEN-RadLlama2-7b",
                        help='Path to GREEN model weights')
    parser.add_argument('--compute-llama-score', action=argparse.BooleanOptionalAction, default=False, help='Compute Llama Score (slow). Use --compute-llama-score to enable.')
    parser.add_argument('--compute-clinical-efficacy', action=argparse.BooleanOptionalAction, default=True, help='Compute clinical efficacy for report_generation. Use --no-compute-clinical-efficacy to skip.')
    parser.add_argument('--llama-model-path', type=str,
                        default="/path/to/Meta-Llama-3-70B-Instruct",
                        help='Path to Llama model for Llama Score')
    parser.add_argument('--radbert-checkpoint', type=str,
                        default="/path/to/CT-CLIP/RadBertClassifier.pth",
                        help='Path to RadBERT checkpoint for clinical efficacy')
    parser.add_argument(
        '--clinical-labels-csv',
        type=str,
        default="data/annotations/valid_predicted_labels.csv",
        help=(
            'Optional: path to CT-RATE multi_abnormality_labels CSV (e.g. train_predicted_labels.csv). '
            'If provided, clinical efficacy will use these labels as ground truth (aligned by prediction["image"] == CSV VolumeName).'
        )
    )
    parser.add_argument('--device', type=str, default='cuda', help='Device for metrics computation')
    parser.add_argument('--bootstrap', type=int, default=0,
                        help='Number of bootstrap iterations for confidence intervals (0=disabled, recommended: 1000)')

    return parser.parse_args()


def _normalize_volume_name(name: str) -> str:
    s = str(name).strip()
    # Keep only filename if a path is provided
    s = Path(s).name
    if s.endswith('.nii.gz'):
        s = s[:-7]
    return s


def load_clinical_labels_csv(csv_path: str) -> Dict[str, List[int]]:
    """Load VolumeName -> 18-dim binary label vector."""
    labels_by_volume: Dict[str, List[int]] = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV or missing header: {csv_path}")

        missing_cols = [c for c in ['VolumeName', *CLINICAL_FINDINGS] if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                f"Clinical labels CSV missing required columns: {missing_cols}. "
                f"Got columns: {reader.fieldnames}"
            )

        for row in reader:
            vol = _normalize_volume_name(row['VolumeName'])
            vec: List[int] = []
            for finding in CLINICAL_FINDINGS:
                v = str(row[finding]).strip()
                if v not in ('0', '1'):
                    try:
                        v_int = int(float(v))
                    except Exception as e:
                        raise ValueError(
                            f"Invalid label value in CSV {csv_path}: VolumeName={row['VolumeName']}, "
                            f"finding={finding}, value={row[finding]}"
                        ) from e
                    v = str(v_int)
                vec.append(1 if v == '1' else 0)
            labels_by_volume[vol] = vec
    return labels_by_volume


def load_predictions(predictions_path: str) -> List[Dict]:
    """Load predictions from JSONL file.
    
    For report_generation tasks, overrides the question field with a unified prompt.
    """
    predictions = []
    with open(predictions_path, 'r') as f:
        for line in f:
            pred = json.loads(line)
            if pred["task_type"] == "report_generation":
                pred["question"] = REPORT_GENERATION_PROMPT
            predictions.append(pred)
    return predictions


def split_by_task(predictions: List[Dict]) -> Dict[str, List[Dict]]:
    """Split predictions by task type."""
    by_task = defaultdict(list)
    for pred in predictions:
        task_type = pred["task_type"]
        by_task[task_type].append(pred)
    return dict(by_task)


def _precompute_bleu_stats(hyps: List[str], refs: List[str], max_n: int = 4):
    """Pre-compute per-sentence n-gram match/total counts for corpus BLEU.

    Returns arrays of shape (N, max_n) for matches and totals,
    and (N,) for ref_lens and hyp_lens.
    """
    from collections import Counter

    n_samples = len(hyps)
    matches = np.zeros((n_samples, max_n), dtype=np.float64)
    totals  = np.zeros((n_samples, max_n), dtype=np.float64)
    ref_lens = np.zeros(n_samples, dtype=np.float64)
    hyp_lens = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        hyp_words = hyps[i].split()
        ref_words = refs[i].split()
        hyp_lens[i] = len(hyp_words)
        ref_lens[i] = len(ref_words)

        for ng in range(1, max_n + 1):
            if len(hyp_words) < ng:
                continue
            hyp_ngrams = Counter(tuple(hyp_words[j:j+ng]) for j in range(len(hyp_words) - ng + 1))
            ref_ngrams = Counter(tuple(ref_words[j:j+ng]) for j in range(len(ref_words) - ng + 1))
            matches[i, ng - 1] = sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
            totals[i, ng - 1]  = sum(hyp_ngrams.values())

    return matches, totals, ref_lens, hyp_lens


def _corpus_bleu_from_aggregated(matches_sum, totals_sum, ref_len_sum, hyp_len_sum, max_n=4):
    """Compute corpus BLEU-1 .. BLEU-max_n from aggregated n-gram counts.

    Returns list [bleu_1, bleu_2, ..., bleu_max_n].
    """
    import math

    if hyp_len_sum == 0:
        return [0.0] * max_n

    bp = min(1.0, math.exp(1.0 - ref_len_sum / hyp_len_sum))

    scores = []
    log_prec_sum = 0.0
    for ng in range(max_n):
        if totals_sum[ng] == 0 or matches_sum[ng] == 0:
            scores.extend([0.0] * (max_n - ng))
            break
        log_prec_sum += math.log(matches_sum[ng] / totals_sum[ng])
        scores.append(bp * math.exp(log_prec_sum / (ng + 1)))

    return scores


def _bootstrap_ci(
    hyps: List[str],
    refs: List[str],
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for NLP and clinical metrics.

    Strategy:
      - BLEU: pre-compute per-sentence n-gram statistics, then for each
        bootstrap sample aggregate counts and compute true corpus-level BLEU
        (preserves brevity penalty & corpus-level precision). Pure numpy + math.
      - ROUGE-L: pre-compute per-sentence scores, bootstrap by resampling
        means (exact, since corpus ROUGE-L = mean of per-sentence ROUGE-L).
      - Clinical F1: recompute from resampled (y_true, y_pred) rows each
        iteration (pure numpy, very fast).

    Args:
        hyps: Generated texts (report_generation).
        refs: Reference texts.
        y_true: Ground-truth clinical labels (N x 18), or None.
        y_pred: Predicted clinical labels (N x 18), or None.
        n_bootstrap: Number of bootstrap iterations.
        alpha: Significance level (0.05 -> 95% CI).

    Returns:
        Dict[metric_name -> {ci_lower, ci_upper, bootstrap_mean, bootstrap_std}].
    """
    from pycocoevalcap.rouge.rouge import Rouge

    rng = np.random.RandomState(42)
    n = len(hyps)

    # Pre-process texts
    hyps_lower = [h.strip().lower() if h.strip() else " " for h in hyps]
    refs_lower = [r.strip().lower() if r.strip() else " " for r in refs]

    # ── Step 1: pre-compute statistics ONCE ──
    print("  Pre-computing BLEU n-gram statistics ...")
    bleu_m, bleu_t, bleu_rl, bleu_hl = _precompute_bleu_stats(hyps_lower, refs_lower)

    # Verify against full-corpus value
    full_bleu = _corpus_bleu_from_aggregated(
        bleu_m.sum(0), bleu_t.sum(0), bleu_rl.sum(), bleu_hl.sum())
    print(f"    Corpus BLEU-1={full_bleu[0]:.4f}, BLEU-4={full_bleu[3]:.4f}")

    print("  Pre-computing ROUGE-L per-sentence scores ...")
    h_dict = {i: [h] for i, h in enumerate(hyps_lower)}
    r_dict = {i: [r] for i, r in enumerate(refs_lower)}
    _, rouge_per = Rouge().compute_score(r_dict, h_dict)
    rouge_arr = np.array(rouge_per)

    has_clinical = y_true is not None and y_pred is not None
    print(f"  Pre-computation done (N={n}). Running {n_bootstrap} bootstrap iterations ...")

    # ── Step 2: bootstrap ──
    all_idx = rng.choice(n, size=(n_bootstrap, n), replace=True)

    boot_bleu1 = np.zeros(n_bootstrap)
    boot_bleu4 = np.zeros(n_bootstrap)
    boot_rouge = np.zeros(n_bootstrap)

    for b, idx in enumerate(all_idx):
        # Corpus BLEU from aggregated n-gram counts (true corpus-level metric)
        m_sum = bleu_m[idx].sum(axis=0)
        t_sum = bleu_t[idx].sum(axis=0)
        rl_sum = bleu_rl[idx].sum()
        hl_sum = bleu_hl[idx].sum()
        bleu_scores = _corpus_bleu_from_aggregated(m_sum, t_sum, rl_sum, hl_sum)
        boot_bleu1[b] = bleu_scores[0]
        boot_bleu4[b] = bleu_scores[3]

        # ROUGE-L: mean of per-sentence (exact)
        boot_rouge[b] = rouge_arr[idx].mean()

    score_arrays = {
        "bleu_1":  boot_bleu1,
        "bleu_4":  boot_bleu4,
        "rouge_l": boot_rouge,
    }

    # Clinical: must recompute (aggregate metric, not per-sample mean)
    if has_clinical:
        boot_cp, boot_cr, boot_cf = [], [], []
        for idx in all_idx:
            clin = compute_clinical_f1(y_true[idx], y_pred[idx], CLINICAL_FINDINGS)
            boot_cp.append(clin["precision"])
            boot_cr.append(clin["recall"])
            boot_cf.append(clin["f1"])
        score_arrays["clinical_precision"] = np.array(boot_cp)
        score_arrays["clinical_recall"]    = np.array(boot_cr)
        score_arrays["clinical_f1"]        = np.array(boot_cf)

    # ── Step 3: compute confidence intervals ──
    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100

    ci_results: Dict[str, Dict[str, float]] = {}
    for key, vals in score_arrays.items():
        ci_results[key] = {
            "ci_lower": round(float(np.percentile(vals, lo_pct)), 6),
            "ci_upper": round(float(np.percentile(vals, hi_pct)), 6),
            "bootstrap_mean": round(float(np.mean(vals)), 6),
            "bootstrap_std": round(float(np.std(vals)), 6),
        }

    return ci_results


def calculate_metrics(
    predictions: List[Dict],
    compute_green: bool = True,
    green_model_path: Optional[str] = None,
    compute_llama: bool = False,
    compute_clinical: bool = False,
    llama_model_path: Optional[str] = None,
    radbert_checkpoint: Optional[str] = None,
    clinical_labels_csv: Optional[str] = None,
    device: str = 'cuda',
    bootstrap: int = 0,
) -> Dict[str, float]:
    """
    Calculate all metrics from predictions.

    Args:
        predictions: List of prediction dicts
        compute_green: Whether to compute GREEN score
        green_model_path: Path to GREEN model
        compute_llama: Whether to compute Llama Score
        compute_clinical: Whether to compute clinical efficacy
        llama_model_path: Path to Llama model
        radbert_checkpoint: Path to RadBERT checkpoint
        device: Device for computation

    Returns:
        OrderedDict with all metrics (GREEN first, then per-task and overall)
    """
    print("=" * 80)
    print("Calculating Metrics")
    print("=" * 80)
    print(f"Total predictions: {len(predictions)}")

    # Split by task
    by_task = split_by_task(predictions)
    print(f"\nPer-task counts:")
    for task in TASK_ORDER:
        if task in by_task:
            print(f"  {task}: {len(by_task[task])}")

    metrics = OrderedDict()
    metrics["num_samples"] = len(predictions)

    # Data for bootstrap CI (populated in sections below)
    _rg_hyps = None
    _rg_refs = None
    _rg_y_true = None
    _rg_y_pred = None

    for task in TASK_ORDER:
        if task in by_task:
            metrics[f"{task}/num_samples"] = len(by_task[task])

    # ── [1/4] NLP text metrics (BLEU, METEOR, ROUGE-L, CIDEr) ──
    print("\n" + "-" * 80)
    print("[1/4] Computing Text Metrics (BLEU, METEOR, ROUGE-L, CIDEr)")
    print("-" * 80)

    task_text_metrics = {}
    for task in TASK_ORDER:
        if task not in by_task or len(by_task[task]) == 0:
            continue

        task_preds = by_task[task]
        hyps = [_extract(p, _HYP_KEYS) for p in task_preds]
        refs = [_extract(p, _REF_KEYS) for p in task_preds]

        print(f"\n{task} ({len(hyps)} samples):")
        task_metrics = compute_all_text_metrics(hyps, refs)

        for k, v in task_metrics.items():
            metrics[f"{task}/{k}"] = v

        task_text_metrics[task] = task_metrics
        if task == "report_generation":
            _rg_hyps = hyps
            _rg_refs = refs

        # Print immediately
        for k, v in sorted(task_metrics.items()):
            print(f"  {k}: {v:.4f}")

    # Overall text metrics (average across tasks)
    if task_text_metrics:
        print(f"\n  --- Overall (avg across {len(task_text_metrics)} tasks) ---")
        metric_keys = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "bleu_sacre", "rouge_l", "meteor", "cider"]
        for key in metric_keys:
            values = [task_text_metrics[task][key] for task in task_text_metrics if key in task_text_metrics[task]]
            if values:
                metrics[f"overall/{key}"] = sum(values) / len(values)
                print(f"  overall/{key}: {metrics[f'overall/{key}']:.4f}")

        if "overall/meteor" in metrics and "overall/rouge_l" in metrics:
            metrics["overall/meteor_rouge"] = (metrics["overall/meteor"] + metrics["overall/rouge_l"]) / 2
            print(f"  overall/meteor_rouge: {metrics['overall/meteor_rouge']:.4f}")

    # ── [2/4] Clinical efficacy (report_generation only) ──
    if compute_clinical and radbert_checkpoint and "report_generation" in by_task:
        print("\n" + "-" * 80)
        print("[2/4] Computing Clinical Efficacy (Report Generation only)")
        print("-" * 80)

        if not clinical_labels_csv:
            raise ValueError(
                "clinical efficacy now requires --clinical-labels-csv (ground-truth 18-labels). "
                "Reference-text-based label extraction has been removed."
            )

        report_preds = by_task["report_generation"]
        hyps = [_extract(p, _HYP_KEYS) for p in report_preds]

        labels_by_volume = load_clinical_labels_csv(clinical_labels_csv)
        y_true = []
        missing = []
        for p in report_preds:
            vol = p['image']
            if vol is None:
                missing.append('<missing image field>')
                continue
            key = _normalize_volume_name(vol)
            if key not in labels_by_volume:
                missing.append(vol)
                continue
            y_true.append(labels_by_volume[key])

        if missing:
            preview = missing[:20]
            hint = ""
            if len(missing) == len(report_preds) and len(report_preds) > 0:
                first_key = str(report_preds[0]['image'].strip())
                first_key = Path(first_key).name
                if first_key.startswith('valid_') and 'train_predicted_labels.csv' in str(clinical_labels_csv):
                    hint = " Hint: your predictions look like valid_* but you are using train_predicted_labels.csv; try valid_predicted_labels.csv."
                if first_key.startswith('train_') and 'valid_predicted_labels.csv' in str(clinical_labels_csv):
                    hint = " Hint: your predictions look like train_* but you are using valid_predicted_labels.csv; try train_predicted_labels.csv."
            raise KeyError(
                f"Failed to align clinical labels: {len(missing)} / {len(report_preds)} samples missing in CSV. "
                f"First missing: {preview}." + hint
            )

        print(f"  Extracting findings from generated reports...")
        _rg_y_pred = extract_findings(hyps, radbert_checkpoint, device)
        _rg_y_true = np.asarray(y_true).astype(int)
        print(f"  y_pred shape: {_rg_y_pred.shape}, y_true shape: {_rg_y_true.shape}")

        clinical_result = compute_clinical_f1(_rg_y_true, _rg_y_pred.astype(int), CLINICAL_FINDINGS)

        metrics["report_generation/clinical_precision"] = clinical_result["precision"]
        metrics["report_generation/clinical_recall"] = clinical_result["recall"]
        metrics["report_generation/clinical_f1"] = clinical_result["f1"]
        metrics["report_generation/clinical_detailed"] = clinical_result["per_finding"]

        # Print per-finding results
        print("\n  Per-finding metrics:")
        print("  " + "-" * 78)
        for _fn, _fd in clinical_result["per_finding"].items():
            print(f"  {_fn:40s} | P: {_fd['precision']:.3f} | R: {_fd['recall']:.3f} | F1: {_fd['f1']:.3f} | Sup: {_fd['support']}")
        print(f"\n  Clinical Precision: {clinical_result['precision']:.4f}")
        print(f"  Clinical Recall:    {clinical_result['recall']:.4f}")
        print(f"  Clinical F1:        {clinical_result['f1']:.4f}")

    # ── [3/4] Llama Score (LLM-as-Judge) ──
    if compute_llama and llama_model_path:
        print("\n" + "-" * 80)
        print("[3/4] Computing Llama Score (LLM-as-Judge)")
        print("-" * 80)

        task_llama_scores = {}
        for task in TASK_ORDER:
            if task not in by_task or len(by_task[task]) == 0:
                continue

            task_preds = by_task[task]
            hyps = [_extract(p, _HYP_KEYS) for p in task_preds]
            refs = [_extract(p, _REF_KEYS) for p in task_preds]
            questions = [_extract(p, _Q_KEYS) for p in task_preds]

            print(f"\n{task} ({len(hyps)} samples):")
            llama_result = compute_llama_score(
                hypotheses=hyps,
                references=refs,
                task_type=task,
                questions=questions,
                model_path=llama_model_path,
            )

            metrics[f"{task}/llama_score"] = llama_result["llama_score"]
            task_llama_scores[task] = llama_result["llama_score"]
            # Print immediately
            print(f"  {task}/llama_score: {llama_result['llama_score']:.4f}")

        if task_llama_scores:
            metrics["overall/llama_score"] = sum(task_llama_scores.values()) / len(task_llama_scores)
            print(f"\n  Overall Llama Score: {metrics['overall/llama_score']:.4f}")

    # ── [4/4] GREEN Score (report_generation only) ──
    if compute_green and green_model_path and "report_generation" in by_task:
        print("\n" + "-" * 80)
        print("[4/4] Computing GREEN Score (report_generation only)")
        print("-" * 80)

        report_preds = by_task["report_generation"]
        hyps = [_extract(p, _HYP_KEYS) for p in report_preds]
        refs = [_extract(p, _REF_KEYS) for p in report_preds]

        print(f"  {len(refs)} samples")
        green_scorer = GREEN(green_model_path, output_dir=".", cpu=False)
        green_scorer.batch_size = 16
        mean, std, _, _, _ = green_scorer(refs, hyps)

        metrics["report_generation/green_mean"] = round(float(mean), 6)
        metrics["report_generation/green_std"] = round(float(std), 6)
        metrics["overall/green_mean"] = round(float(mean), 6)
        metrics["overall/green_std"] = round(float(std), 6)

        # Print immediately
        print(f"  GREEN Score: {mean:.4f} ± {std:.4f}")

    # ── Bootstrap Confidence Intervals ──
    if bootstrap > 0 and _rg_hyps is not None:
        print("\n" + "-" * 80)
        print(f"Computing Bootstrap Confidence Intervals ({bootstrap} iterations)")
        print("-" * 80)

        ci = _bootstrap_ci(
            hyps=_rg_hyps,
            refs=_rg_refs,
            y_true=_rg_y_true,
            y_pred=_rg_y_pred,
            n_bootstrap=bootstrap,
        )

        for metric_name, ci_data in ci.items():
            prefix = "report_generation"
            metrics[f"{prefix}/{metric_name}_ci_lower"] = ci_data["ci_lower"]
            metrics[f"{prefix}/{metric_name}_ci_upper"] = ci_data["ci_upper"]
            metrics[f"{prefix}/{metric_name}_bootstrap_mean"] = ci_data["bootstrap_mean"]
            metrics[f"{prefix}/{metric_name}_bootstrap_std"] = ci_data["bootstrap_std"]

        print("\n  95% Confidence Intervals:")
        for metric_name, ci_data in ci.items():
            print(f"    {metric_name:25s}: [{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}] (std={ci_data['bootstrap_std']:.4f})")

    # ── Final summary ──
    print("\n" + "=" * 80)
    print("All Metrics Summary")
    print("=" * 80)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif not isinstance(v, dict):
            print(f"  {k}: {v}")

    print("=" * 80)

    return metrics


def save_metrics(metrics: Dict[str, float], output_path: str):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_path}")


def main():
    """Main metrics calculation pipeline"""
    args = parse_args()

    print("=" * 80)
    print("Stage 3 Turn Metrics Calculation")
    print("=" * 80)

    # Load predictions
    print(f"\nLoading predictions from: {args.predictions}")
    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} predictions")

    # Calculate metrics
    metrics = calculate_metrics(
        predictions=predictions,
        compute_green=args.compute_green,
        green_model_path=args.green_model_path,
        compute_llama=args.compute_llama_score,
        compute_clinical=args.compute_clinical_efficacy,
        llama_model_path=args.llama_model_path,
        radbert_checkpoint=args.radbert_checkpoint,
        clinical_labels_csv=args.clinical_labels_csv,
        device=args.device,
        bootstrap=args.bootstrap,
    )

    # Save metrics
    if args.output:
        save_metrics(metrics, args.output)
    else:
        # Generate output path from predictions path
        try:
            output_path = generate_metrics_path_from_predictions(args.predictions)
            save_metrics(metrics, str(output_path))
        except ValueError as e:
            print(f"Error: Failed to parse predictions path: {e}")
            print("Please ensure predictions file is in format: .../infer/infer_step_<number>.jsonl")
            return None

    return metrics


if __name__ == '__main__':
    main()
