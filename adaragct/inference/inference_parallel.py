"""
Parallel RAG Token Inference: N model replicas on a single GPU.

Loads N copies of the model on one GPU (192GB B200 fits 3× 8B fp16 easily).
Each worker processes 1/N of the samples concurrently via torch.multiprocessing.

Usage:
    python P2_rag/inference/inference_rag_token_parallel.py \
        --checkpoint results/.../checkpoints/step_2000 --num-workers 3

    # With options (same as inference_rag_token.py):
    python P2_rag/inference/inference_rag_token_parallel.py \
        --checkpoint ... --num-workers 3 --oracle --max-samples 100
"""

import argparse
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import torch
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from adaragct.inference.inference_rag import (
    load_model,
    load_validation_data,
    build_context_index,
    lookup_context,
    load_text_encoder,
    load_sentence_db,
    load_img2img_top20,
    online_retrieve_context,
    rag_token_generate,
    IMG2IMG_WHOLE_CT_PATH,
    SENTENCE_DB_PATH,
)
from adaragct.utils.rag_utils import RET_START_TOKEN, RET_END_TOKEN, RAG_TOKEN
from adaragct.data.dataset import (
    ORGAN_TOKENS_DESC, SYSTEM_PROMPTS, _build_chat_prompt, TASK_TOKENS,
)


def worker_fn(
    worker_id: int,
    checkpoint: str,
    config: dict,
    sample_indices: List[int],
    all_samples: list,
    prompt: str,
    mode: str,
    args_dict: dict,
    global_context_map: Optional[dict],
    output_path: str,
    write_lock,
):
    """Worker process: load model, process assigned samples, write one line per sample immediately."""
    device = torch.device("cuda")
    tag = f"[W{worker_id}]"

    print(f"{tag} Loading model...")
    model, tokenizer, ret_start_id, ret_end_id, rag_token_id = load_model(checkpoint, device)

    # Load RAG resources per worker
    context_index = None
    all_context_texts = None
    rag_resources = None

    ORACLE_CONTEXT_PATH = "data/oracle/oracle_context_top3.jsonl"

    if mode == "oracle":
        context_index = build_context_index(ORACLE_CONTEXT_PATH)
        print(f"{tag} Oracle context loaded ({len(context_index)} entries)")
    elif mode == "normal_rag":
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
        print(f"{tag} Normal RAG resources loaded")
    elif mode == "corrupt_rag":
        sentence_data, sentence_embeds, sid_to_idx, sid_to_sent = load_sentence_db()
        all_context_texts = [s["text"] for sents in sentence_data.values() for s in sents]
        rag_resources = {
            "e2e_model": None, "img2img_top20": None,
            "sentence_data": sentence_data, "sentence_embeds": sentence_embeds,
            "sid_to_idx": sid_to_idx,
        }
        print(f"{tag} Corrupt RAG: random pool = {len(all_context_texts)} sentences")

    print(f"{tag} Processing {len(sample_indices)} samples...")
    results = []
    total_retrievals = 0

    for local_i, si in enumerate(sample_indices):
        sample = all_samples[si]
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
            max_new_tokens=args_dict["max_new_tokens"],
            no_rag=(mode == "no_rag"),
            corrupt_rag=(mode == "corrupt_rag"),
            all_context_texts=all_context_texts,
            max_retrievals=args_dict["max_retrievals"],
            cooldown_tokens=args_dict["cooldown_tokens"],
            mode=mode,
            rag_resources=rag_resources,
            rag_logit_bias=args_dict["rag_logit_bias"],
            top_k=args_dict["top_k"],
            retrieval_stage=args_dict["retrieval_stage"],
        )

        total_retrievals += out["retrieval_count"]
        result = {
            "image":           sample["image_id"],
            "generated_text":  out["generated_text"],
            "reference_text":  sample["raw_report"],
            "task_type":       "report_generation",
            "retrieval_count": out["retrieval_count"],
            "retrieval_log":   out["retrieval_log"],
            "rag_logit_bias":  args_dict["rag_logit_bias"],
            "cooldown_tokens": args_dict["cooldown_tokens"],
            "global_context_words": gc_words,
            "_original_index": si,  # for ordering
        }
        results.append(result)

        # Write one line immediately so partial results survive if job is killed
        with write_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")

        print(f"{tag} [{local_i+1}/{len(sample_indices)}] "
              f"image={sample['image_id']} | retrievals={out['retrieval_count']}")

    avg_ret = total_retrievals / len(results) if results else 0
    print(f"{tag} Done. {len(results)} samples, avg_retrievals={avg_ret:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Parallel RAG Token Inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--corrupt-rag", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--max-retrievals", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--rag-logit-bias", type=float, default=0.0)
    parser.add_argument("--cooldown-tokens", type=int, default=5)
    parser.add_argument("--output-suffix", type=str, default=None)
    parser.add_argument("--global-context", action="store_true")
    parser.add_argument("--global-context-topk", type=int, default=3)
    parser.add_argument("--text2text", action="store_true",
                        help="One-stage retrieval: direct text-to-text (skip img2img Stage 1)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-workers", type=int, default=3,
                        help="Number of model replicas on the same GPU (default: 3)")
    args = parser.parse_args()

    # Load config
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
    else:
        raise FileNotFoundError(f"No training_state.pt in {ckpt_path} and no --config provided.")

    # Determine mode
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
            mode = "normal_rag"
    print(f"Mode: {mode}" + (" (text2text one-stage)" if args.text2text else "") + f" | Workers: {args.num_workers}")

    # Auto-infer output path
    if args.output is None:
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

    # Load validation data (lightweight, in main process)
    print("Loading validation data...")
    samples = load_validation_data(config, max_samples=args.max_samples)
    print(f"  {len(samples)} samples")

    # Build prompt
    system_prompt = SYSTEM_PROMPTS[config["data"]["system_prompt"]]
    user_content = (
        f"{ORGAN_TOKENS_DESC}\n{TASK_TOKENS['report_generation']}\n"
        "Would you mind generating the radiology report for the specified chest CT scan?"
    )
    (prompt,) = _build_chat_prompt(system_prompt, user_content)

    # Global context (built in main process — small data)
    global_context_map = None
    if args.global_context:
        print(f"[GLOBAL] Loading global context (top-{args.global_context_topk})...")
        with open(IMG2IMG_WHOLE_CT_PATH) as f:
            img2img_whole_ct = json.load(f)
        with open(SENTENCE_DB_PATH) as f:
            gc_sentence_db = json.load(f)
        report_index = {img_id: " ".join(s["text"] for s in sents)
                        for img_id, sents in gc_sentence_db.items()}
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

    # Split samples across workers (round-robin for balanced load)
    num_workers = min(args.num_workers, len(samples))
    worker_indices = [[] for _ in range(num_workers)]
    for i in range(len(samples)):
        worker_indices[i % num_workers].append(i)

    for w in range(num_workers):
        print(f"  Worker {w}: {len(worker_indices[w])} samples")

    # Single output file: workers append one line per sample (with lock). No temp files.
    write_lock = mp.Manager().Lock()
    # Truncate or create so we start fresh; workers will append
    with open(output_path, "w") as f:
        pass

    args_dict = {
        "max_new_tokens": args.max_new_tokens,
        "max_retrievals": args.max_retrievals,
        "cooldown_tokens": args.cooldown_tokens,
        "rag_logit_bias": args.rag_logit_bias,
        "top_k": args.top_k,
        "retrieval_stage": args.retrieval_stage,  # "one" if --text2text, else "two"
    }

    # Spawn workers
    print(f"\nSpawning {num_workers} workers...")
    t0 = time.time()

    mp.set_start_method("spawn", force=True)
    processes = []
    for w in range(num_workers):
        p = mp.Process(
            target=worker_fn,
            args=(
                w,
                args.checkpoint,
                config,
                worker_indices[w],
                samples,
                prompt,
                mode,
                args_dict,
                global_context_map,
                str(output_path),
                write_lock,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t0

    # Check for failures
    for w, p in enumerate(processes):
        if p.exitcode != 0:
            print(f"[ERROR] Worker {w} exited with code {p.exitcode}")

    # Re-read output (lines may be interleaved), sort by original index, rewrite
    all_results = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                all_results.append(json.loads(line))
    all_results.sort(key=lambda x: x["_original_index"])
    for r in all_results:
        del r["_original_index"]
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    total_ret = sum(r["retrieval_count"] for r in all_results)
    avg_ret = total_ret / len(all_results) if all_results else 0

    print(f"\n{'='*60}")
    print(f"Saved {len(all_results)} predictions → {output_path}")
    print(f"Avg retrievals per sample: {avg_ret:.2f}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/len(all_results):.2f}s/sample)")
    print(f"Workers: {num_workers} | Speedup vs serial: ~{num_workers}x theoretical")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
