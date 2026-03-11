"""
Stage 3 Turn Batch Inference

Batch inference script for single-turn VQA tasks using:
1. LoRA merging for faster inference
2. Left-padding for batch generation
3. Projector-aware embedding projection (shared / independent / grouped)

Usage:
    python stage3_turn/inference/predict_batch.py \
        --peft-checkpoint path/to/checkpoints/step_XXXX \
        --batch-size 8 \
        --max-samples 100
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaragct.utils.seed import set_seed
from adaragct.utils.logger import ConsoleLogger
from adaragct.utils.path_utils import (
    apply_default_values,
    apply_cli_overrides,
)
from adaragct.utils.io import save_jsonl, load_jsonl

from adaragct.data.dataset import (
    ViSDVQADataset,
    collate_fn_vqa,
    NUM_EMBEDDINGS,
)
from adaragct.models.build_model import load_model_peft
from adaragct.train.train_step import project_embeddings
from adaragct.utils.tokenizer_utils import tokenizer_organ_token


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 Turn Batch Inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PEFT folder checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output JSONL path')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for generation')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to process')
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--repetition-penalty', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--sample-types', type=str, default=None,
                        help='JSON string, e.g. \'["image", "paper"]\'')
    parser.add_argument('--label-csv', type=str, default=None,
                        help='Path to label CSV for D5.2 Label-in-Prompt inference (e.g. valid_predicted_labels.csv)')
    return parser.parse_args()


def build_batch_inference_config_peft(peft_dir: str, args) -> Dict:
    """Load config from PEFT folder's training_state.pt, apply defaults and CLI overrides."""
    import json
    training_state_path = Path(peft_dir) / "training_state.pt"
    adapter_config_path = Path(peft_dir) / "adapter_config.json"
    if training_state_path.exists():
        state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        base_config = state.get("config", {})
    elif adapter_config_path.exists():
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)
        base_config = {}
    else:
        raise FileNotFoundError(f"No config found in PEFT dir: {peft_dir}")
    config = apply_default_values(base_config)
    config = apply_cli_overrides(config, args)
    config["model"]["use_flash_attn"] = False
    return config


@torch.no_grad()
def generate_batch(
    model: nn.Module,
    tokenizer,
    batch_embeddings: torch.Tensor,
    batch_prompts: List[str],
    batch_has_images: List[bool],
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    repetition_penalty: float = 1.1,
    top_p: float = 0.9,
    top_k: int = 50,
) -> List[str]:
    """Generate text for a batch of samples.

    Args:
        batch_embeddings: (B, num_embeddings, embed_dim) tensor
        batch_prompts: List of prompt strings with organ tokens
        batch_has_images: List of bools indicating image samples
        max_new_tokens: Shared max_new_tokens for the entire batch

    Returns:
        List of generated text strings.
    """
    batch_size = len(batch_prompts)

    # 1. Project embeddings through projector (handles shared/independent/grouped)
    images_list = project_embeddings(
        embeddings=batch_embeddings,
        model=model,
        has_images=batch_has_images,
        device=device,
    )

    # 2. Tokenize each prompt
    input_ids_list = []
    for prompt in batch_prompts:
        ids = tokenizer_organ_token(
            prompt=prompt.rstrip(),
            tokenizer=tokenizer,
            return_tensors='pt',
            add_special_tokens=True,
        )
        input_ids_list.append(ids)

    # 3. Left-pad to same length
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(ids.shape[0] for ids in input_ids_list)

    padded_ids = []
    attention_masks = []
    for ids in input_ids_list:
        pad_len = max_len - ids.shape[0]
        padded = torch.cat([
            torch.full((pad_len,), pad_id, dtype=torch.long),
            ids,
        ])
        mask = torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.ones(ids.shape[0], dtype=torch.long),
        ])
        padded_ids.append(padded)
        attention_masks.append(mask)

    input_ids = torch.stack(padded_ids).to(device)
    attention_mask = torch.stack(attention_masks).to(device)

    # 4. Generate
    with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
        output_ids = model.generate(
            inputs=input_ids,
            images=images_list,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            top_p=top_p if temperature > 0 else None,
            top_k=top_k if temperature > 0 else None,
            use_cache=True,
        )

    # 5. Decode
    results = []
    for i in range(batch_size):
        text = tokenizer.decode(output_ids[i], skip_special_tokens=True).strip()
        results.append(text)

    return results


def main():
    args = parse_args()
    print("=" * 80)
    print(f"Stage 3 Turn Batch Inference (batch_size={args.batch_size})")
    print("=" * 80)

    # 1. Config
    config = build_batch_inference_config_peft(args.checkpoint, args)
    set_seed(config['experiment']['seed'])
    device = torch.device(config['device']['type'])
    logger = ConsoleLogger()

    # 2. Model
    print("\n[1/4] Building model...")
    from adaragct.utils.logger import set_module_logger
    set_module_logger(logger)
    model, tokenizer = load_model_peft(
        checkpoint_dir=args.checkpoint,
        device=device,
        merge_lora=True,
    )
    print("  Loaded via load_model_peft() (PEFT folder format)")

    model.eval()
    model.config.tokenizer_model_max_length = 2048

    # Left-padding for batch generation
    tokenizer.padding_side = "left"
    model.config.tokenizer_padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generation config
    gen_config = {
        "temperature": config["generation"]["temperature"],
        "repetition_penalty": config["generation"]["repetition_penalty"],
        "top_p": config["generation"]["top_p"],
        "top_k": config["generation"]["top_k"],
    }
    print(f"  Generation config: {gen_config}")

    # 3. Dataset
    print("\n[2/4] Building dataset...")
    train_tasks = config["data"]["train_tasks"]
    whole_ct_key = config["data"]["whole_ct_key"]
    organ_keys = config["data"]["organ_keys"]
    embedding_keys = [whole_ct_key] + organ_keys

    sample_types = config["data"].get("sample_types")
    if sample_types is None:
        raise ValueError("sample_types required in config[data]")

    use_chat_template = config["data"].get("use_chat_template", False)
    system_prompt = config["data"].get("system_prompt", "minimal")
    whole_ct_npz_val = config["data"].get("whole_ct_embeddings_npz_val", None)
    whole_ct_dim = config["data"].get("whole_ct_dim", None)
    # D5.2 Label-in-Prompt: use --label-csv if provided, else fall back to config
    label_csv_path = args.label_csv or config["data"].get("label_csv_path", None)
    label_in_prompt = config["data"].get("label_in_prompt", False)
    if args.label_csv:
        label_in_prompt = True
        print(f"  D5.2 Label-in-Prompt: using label CSV {label_csv_path}")

    val_dataset = ViSDVQADataset(
        embeddings_npz=config["data"]["val_embeddings_npz"],
        vqa_json_path=config["data"]["val_vqa_json"],
        train_tasks=train_tasks,
        embedding_keys=embedding_keys,
        sample_types=sample_types,
        use_chat_template=use_chat_template,
        system_prompt=system_prompt,
        whole_ct_npz=whole_ct_npz_val,
        whole_ct_dim=whole_ct_dim,
        label_in_prompt=label_in_prompt,
        label_csv_path=label_csv_path,
    )

    # Load already-completed predictions early so we can filter the dataset
    if args.output is None:
        ckpt_dir = Path(args.checkpoint)
        output_file = str(ckpt_dir.parent.parent / "infer" / f"infer_{ckpt_dir.name}.jsonl")
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = args.output
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    existing_predictions = []
    done_keys = set()
    if Path(output_file).exists():
        existing_predictions = load_jsonl(output_file)
        for p in existing_predictions:
            done_keys.add((p["study_id"], p["task_type"]))
        print(f"  Resuming: {len(done_keys)} samples already done, skipping them.")

    # Build index list: skip already-done samples
    all_indices = list(range(len(val_dataset)))
    if done_keys:
        pending_indices = []
        for idx in all_indices:
            sample = val_dataset[idx]
            key = (sample["sample_id"], sample["task_type"])
            if key not in done_keys:
                pending_indices.append(idx)
    else:
        pending_indices = all_indices

    if args.max_samples is not None:
        remaining = args.max_samples - len(done_keys)
        pending_indices = pending_indices[:max(0, remaining)]

    val_dataset_for_loader = Subset(val_dataset, pending_indices)

    print(f"  Samples: {len(val_dataset)} total, {len(val_dataset_for_loader)} pending")

    val_loader = DataLoader(
        val_dataset_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=False,
        collate_fn=collate_fn_vqa,
    )

    # Max new tokens per task
    max_new_tokens_by_task = config["validation"]["max_new_tokens_by_task"]

    # 4. Inference
    print(f"\n[3/4] Running inference ({len(val_loader)} batches)...")
    predictions = []
    num_processed = 0
    out_f = open(output_file, 'a')

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Inference")):
        embeddings = batch["embeddings"]
        prompts = batch["prompt"]
        targets = batch["target"]
        raw_targets = batch.get("raw_target", targets)
        task_types = batch["task_type"]
        sample_ids = batch["sample_id"]
        images = batch["image"]
        has_images = batch["has_image"]
        questions = batch["question"]
        batch_size = len(prompts)

        # Determine max_new_tokens for this batch
        batch_tasks = set(task_types)
        fallback = max(max_new_tokens_by_task.values())
        if len(batch_tasks) == 1:
            task = list(batch_tasks)[0]
            max_new = max_new_tokens_by_task.get(task, fallback)
        else:
            max_new = max(max_new_tokens_by_task.get(t, fallback) for t in batch_tasks)

        generated_texts = generate_batch(
            model=model,
            tokenizer=tokenizer,
            batch_embeddings=embeddings,
            batch_prompts=prompts,
            batch_has_images=has_images,
            device=device,
            max_new_tokens=max_new,
            **gen_config,
        )

        batch_new = []
        for i in range(batch_size):
            key = (sample_ids[i], task_types[i])
            if key in done_keys:
                continue
            record = {
                "study_id": sample_ids[i],
                "task_type": task_types[i],
                "generated_text": generated_texts[i],
                "reference_text": raw_targets[i],
                "input_prompt": prompts[i],
                "question": questions[i] if task_types[i] != "report_generation" else prompts[i],
                "image": images[i],
            }
            batch_new.append(record)
            predictions.append(record)
            num_processed += 1
            if args.max_samples is not None and num_processed >= args.max_samples:
                break

        # Immediately flush new records to disk
        for record in batch_new:
            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
        out_f.flush()

        # Print last sample of batch for quality check
        if predictions:
            last = predictions[-1]
            print(f"\n{'─'*60}")
            print(f"[{last['task_type']}] {last['study_id']}")
            print(f"Generated: {last['generated_text'][:200]}")
            print(f"Reference: {last['reference_text'][:200]}")
            print(f"{'─'*60}")

        if args.max_samples is not None and num_processed >= args.max_samples:
            break

    out_f.close()
    print(f"\n[4/4] Done. {len(existing_predictions)} existing + {len(predictions)} new = {len(existing_predictions) + len(predictions)} total predictions.")

    # Summary
    print("\n" + "=" * 80)
    task_counts = defaultdict(int)
    for pred in predictions:
        task_counts[pred["task_type"]] += 1
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")
    print(f"Output: {output_file}")
    print("=" * 80)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output_file


if __name__ == '__main__':
    main()
