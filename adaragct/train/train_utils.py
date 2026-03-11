"""
Step 2.4: Oracle Training Loop.

Supports trigger strategies: "entropy" and "c5_head".
For [RAG] token training (strategy "rag_token"), use train_rag_token.py instead.

Reuses stage3_turn infrastructure with extensions:
- Loss masking for <|ret_start|>...<|ret_end|> tokens
- Optional C5 head + auxiliary loss (Strategy B)
- Special token registration: <|ret_start|>, <|ret_end|>
- Loads from base model checkpoint (E29 or organ model)

Usage:
    python P2_rag/oracle/train.py --config P2_rag/oracle/configs/R01_raw_entropy.yaml
"""

import argparse
import json
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from adaragct.utils.seed import set_seed
from adaragct.utils.logger import get_module_logger, ConsoleLogger, create_logger, set_module_logger

from adaragct.models.build_model import build_model
from adaragct.train.train_step import project_embeddings
from adaragct.utils.tokenizer_utils import tokenizer_organ_token
from adaragct.constants import ORGAN_TOKENS, ORGAN_TOKEN_NAMES, PROVIDED_TOKEN

from llava.constants import IGNORE_INDEX

from adaragct.data.oracle_dataset import OracleTrainDataset, collate_fn_oracle, RET_START_TOKEN, RET_END_TOKEN, RAG_TOKEN
from adaragct.train.loss import mask_retrieval_context_batch, count_masked_tokens
from adaragct.train.c5_head import C5Head, find_sentence_boundary_positions


def load_config(config_path: str) -> dict:
    """Load YAML config, resolving {datetime} in output_dir."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Resolve {datetime} placeholder
    dt_str = time.strftime("%Y%m%d_%H%M%S")
    output_dir = config["paths"]["output_dir"]
    if "{datetime}" in output_dir:
        config["paths"]["output_dir"] = output_dir.replace("{datetime}", dt_str)
    return config


def build_model_and_tokenizer(config: dict, device: torch.device):
    """Build model, tokenizer, and register special tokens."""
    logger = get_module_logger()

    # Build base model with LoRA
    model, tokenizer = build_model(
        config=config,
        device=device,
        pretrain_checkpoint=config["model"]["pretrain_checkpoint"],
    )

    trigger_strategy = config["oracle"]["trigger_strategy"]
    assert trigger_strategy in ("entropy", "c5_head"), (
        f"trigger_strategy='{trigger_strategy}' is not supported in train.py. "
        "Use train_rag_token.py for 'rag_token'."
    )

    # Register retrieval boundary tokens
    rag_tokens = [RET_START_TOKEN, RET_END_TOKEN]
    new_tokens = [t for t in rag_tokens if t not in tokenizer.get_vocab()]
    if new_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        logger.info(f"Added {len(new_tokens)} RAG tokens: {new_tokens}")
        logger.info(f"Final vocab size: {len(tokenizer)}")

    ret_start_id = tokenizer.convert_tokens_to_ids(RET_START_TOKEN)
    ret_end_id = tokenizer.convert_tokens_to_ids(RET_END_TOKEN)
    logger.info(f"ret_start_id={ret_start_id}, ret_end_id={ret_end_id}")

    # Build C5 head if needed
    c5_head = None
    if trigger_strategy == "c5_head":
        hidden_size = model.config.hidden_size
        c5_head = C5Head(hidden_size).to(device=device, dtype=torch.bfloat16)
        logger.info(f"C5 Head initialized: hidden_size={hidden_size}")

    return model, tokenizer, c5_head


def oracle_train_step(
    model: nn.Module,
    batch: Dict[str, Any],
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    step: int,
    config: dict,
    c5_head: nn.Module = None,
    c5_optimizer: torch.optim.Optimizer = None,
    ret_start_id: int = None,
    ret_end_id: int = None,
    should_print: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """Single oracle training step with loss masking and optional C5 head."""
    model.train()
    if c5_head is not None:
        c5_head.train()

    oracle_config = config["oracle"]
    use_chat_template = config["data"]["use_chat_template"]
    max_seq_length = config["training"]["max_seq_length_by_task"]["report_generation"]
    grad_accum_steps = config["training"]["grad_accum_steps"]
    c5_loss_weight = oracle_config["c5_loss_weight"]

    embeddings = batch["embeddings"].to(device)
    batch_size = embeddings.shape[0]
    prompts = batch["prompt"]
    targets = batch["target"]

    full_ids_list = []
    labels_list = []
    prompt_lens = []

    for i in range(batch_size):
        prompt_text = prompts[i]
        target_text = targets[i]

        # Tokenize prompt
        prompt_ids = tokenizer_organ_token(
            prompt=prompt_text,
            tokenizer=tokenizer,
            add_special_tokens=True,
        )

        # Tokenize target
        if use_chat_template:
            target_with_eos = target_text  # already has <|eot_id|>
        else:
            target_with_eos = target_text + tokenizer.eos_token

        target_ids = tokenizer_organ_token(
            prompt=target_with_eos,
            tokenizer=tokenizer,
            add_special_tokens=False,
        )
        if target_ids and target_ids[0] == tokenizer.bos_token_id:
            target_ids = target_ids[1:]

        full_ids = prompt_ids + target_ids
        prompt_len = len(prompt_ids)

        # Labels: IGNORE_INDEX for prompt, actual IDs for target
        labels = [IGNORE_INDEX] * prompt_len + target_ids

        # Truncate
        if len(full_ids) > max_seq_length:
            full_ids = full_ids[:max_seq_length]
            labels = labels[:max_seq_length]

        full_ids_list.append(full_ids)
        labels_list.append(labels)
        prompt_lens.append(prompt_len)

    # Pad
    max_seq_len = max(len(ids) for ids in full_ids_list)
    pad_token_id = tokenizer.pad_token_id
    full_ids_padded, labels_padded, attention_mask = [], [], []
    for full_ids, labels in zip(full_ids_list, labels_list):
        seq_len = len(full_ids)
        padding_len = max_seq_len - seq_len
        full_ids_padded.append(full_ids + [pad_token_id] * padding_len)
        labels_padded.append(labels + [IGNORE_INDEX] * padding_len)
        attention_mask.append([1] * seq_len + [0] * padding_len)

    full_ids_t = torch.tensor(full_ids_padded, dtype=torch.long, device=device)
    labels_t = torch.tensor(labels_padded, dtype=torch.long, device=device)
    attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=device)

    # Apply retrieval context masking
    if ret_start_id is not None and ret_end_id is not None:
        labels_t = mask_retrieval_context_batch(
            labels_t, full_ids_t, ret_start_id, ret_end_id
        )

    # Project embeddings
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        images_list = project_embeddings(
            embeddings=embeddings,
            model=model,
            has_images=batch["has_image"],
            device=device,
        )
        # Forward pass
        need_hidden = (c5_head is not None)
        outputs = model(
            input_ids=full_ids_t,
            attention_mask=attention_mask_t,
            labels=labels_t,
            images=images_list,
            output_hidden_states=need_hidden,
        )

        lm_loss = outputs.loss / grad_accum_steps

        # C5 head auxiliary loss
        c5_loss_val = 0.0
        c5_info = {}
        if c5_head is not None and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states[-1]  # Last layer: (B, seq, hidden)

            # Find sentence boundary positions and labels
            boundary_positions = []
            boundary_labels = []
            for i in range(batch_size):
                positions = find_sentence_boundary_positions(
                    full_ids_t[i], tokenizer, prompt_lens[i],
                    ret_start_id=ret_start_id,
                )
                boundary_positions.append(positions)

                # Get C5 labels from batch
                c5_label_list = batch["c5_labels"][i]
                # Align: we have more positions than labels typically
                # Use needs_rag from c5_labels in order
                labels_for_positions = []
                label_idx = 0
                for pos in positions:
                    if label_idx < len(c5_label_list):
                        labels_for_positions.append(
                            1 if c5_label_list[label_idx].get("needs_rag", False) else 0
                        )
                        label_idx += 1
                    else:
                        labels_for_positions.append(0)
                boundary_labels.append(labels_for_positions)

            c5_loss, c5_info = c5_head(
                hidden_states, boundary_positions, boundary_labels
            )

            if c5_loss is not None:
                c5_loss_val = c5_loss.item()
                total_loss = lm_loss + c5_loss_weight * c5_loss / grad_accum_steps
            else:
                total_loss = lm_loss
        else:
            total_loss = lm_loss

    # Backward
    total_loss.backward()

    metrics = {
        "loss": lm_loss.item() * grad_accum_steps,
        "total_loss": total_loss.item() * grad_accum_steps,
    }

    if c5_head is not None:
        metrics["c5_loss"] = c5_loss_val
        metrics["c5_accuracy"] = c5_info.get("accuracy", 0.0)
        metrics["c5_n_boundaries"] = c5_info.get("n_boundaries", 0)

    # Mask stats (for logging)
    if should_print:
        mask_stats = count_masked_tokens(labels_t)
        metrics["mask_ratio"] = mask_stats["mask_ratio"]
        metrics["active_tokens"] = mask_stats["active"]

    # Optimizer step
    is_accum_step = (step + 1) % grad_accum_steps == 0
    if is_accum_step:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        metrics["grad_norm"] = grad_norm.item()

        optimizer.step()
        optimizer.zero_grad()

        if c5_head is not None and c5_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(c5_head.parameters(), max_norm=1.0)
            c5_optimizer.step()
            c5_optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()
            metrics["lr"] = scheduler.get_last_lr()[0]

    return total_loss.item() * grad_accum_steps, metrics


def save_checkpoint(
    model, tokenizer, optimizer, scheduler, step, metrics, config,
    output_dir, c5_head=None, c5_optimizer=None,
):
    """Save training checkpoint in PEFT folder format."""
    import json
    from safetensors.torch import save_file as save_safetensors

    ckpt_dir = Path(output_dir) / "checkpoints" / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    sd = model.state_dict()
    model_cfg = config["model"]
    # Support both oracle (old) and rag_token (P03/P04) config layouts
    rag_token_cfg = config.get("rag_token", {})
    oracle_cfg = config.get("oracle", {})
    trigger_strategy = rag_token_cfg.get("trigger_strategy",
                        oracle_cfg.get("trigger_strategy", "rag_token"))

    # Determine RAG tokens saved
    rag_tokens = ["<|ret_start|>", "<|ret_end|>", "[RAG]"]

    # 1. adapter_config.json
    adapter_cfg = {
        "base_model_name_or_path": model_cfg["model_path"],
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": model_cfg["lora_alpha"],
        "lora_dropout": model_cfg["lora_dropout"],
        "modules_to_save": ["embed_tokens", "lm_head"],
        "peft_type": "LORA",
        "r": model_cfg["lora_r"],
        "target_modules": model_cfg.get(
            "lora_target_modules",
            ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"],
        ),
        "task_type": "CAUSAL_LM",
        "_stage3_turn": {
            "projector_type": model_cfg.get("projector_type", "independent"),
            "projector_layers": model_cfg.get("projector_layers", 2),
            "tokens_per_embedding": model_cfg.get("tokens_per_embedding", 1),
            "whole_ct_dim": config.get("data", {}).get("whole_ct_dim", None),
        },
        "_rag": {
            "trigger_strategy": trigger_strategy,
            "rag_tokens": rag_tokens,
        },
    }
    with open(ckpt_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_cfg, f, indent=2)

    # 2. adapter_model.safetensors (LoRA + embed_tokens + lm_head)
    adapter_weights = {}
    for k, v in sd.items():
        if "lora_A" in k or "lora_B" in k or "embed_tokens" in k or "lm_head" in k:
            adapter_weights[k] = v.contiguous().to(torch.float32) if v.dtype == torch.bfloat16 else v.contiguous()
    save_safetensors(adapter_weights, str(ckpt_dir / "adapter_model.safetensors"))

    # 3. projector.pt
    proj_weights = {}
    for k, v in sd.items():
        if "organ_projectors" in k:
            rel_key = k[k.find("organ_projectors.") + len("organ_projectors."):]
            proj_weights[rel_key] = v.detach().clone()
        elif "mm_projector" in k and "organ_projectors" not in k:
            rel_key = k[k.find("mm_projector.") + len("mm_projector."):]
            proj_weights[rel_key] = v.detach().clone()
    if proj_weights:
        torch.save(proj_weights, ckpt_dir / "projector.pt")

    # 4. tokenizer
    tokenizer.save_pretrained(str(ckpt_dir))

    # 5. training_state.pt
    training_state = {
        "step": step,
        "metrics": metrics,
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if scheduler is not None:
        training_state["scheduler_state_dict"] = scheduler.state_dict()
    if c5_head is not None:
        training_state["c5_head_state_dict"] = c5_head.state_dict()
    if c5_optimizer is not None:
        training_state["c5_optimizer_state_dict"] = c5_optimizer.state_dict()
    torch.save(training_state, ckpt_dir / "training_state.pt")

    # 6. Cleanup old checkpoints
    keep_last_n = config.get("checkpoint", {}).get("keep_last_n", None)
    if keep_last_n is not None:
        import shutil
        parent = ckpt_dir.parent
        existing = sorted(
            [d for d in parent.iterdir() if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        for old_dir in existing[:-keep_last_n]:
            shutil.rmtree(old_dir)

    return str(ckpt_dir)


def main():
    parser = argparse.ArgumentParser(description="Oracle Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = create_logger(config)
    set_module_logger(logger)

    # Setup
    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = torch.device(config["device"]["type"])

    oracle_config = config["oracle"]
    trigger_strategy = oracle_config["trigger_strategy"]
    base_model_type = oracle_config["base_model_type"]

    if trigger_strategy == "rag_token":
        raise SystemExit(
            "trigger_strategy='rag_token' is not supported in train.py. "
            "Use train_rag_token.py with a P03/P04 config instead."
        )

    logger.info(f"{'='*70}")
    logger.info(f"Oracle Training: {base_model_type} + {trigger_strategy}")
    logger.info(f"{'='*70}")

    # Build model
    model, tokenizer, c5_head = build_model_and_tokenizer(config, device)

    ret_start_id = tokenizer.convert_tokens_to_ids(RET_START_TOKEN)
    ret_end_id = tokenizer.convert_tokens_to_ids(RET_END_TOKEN)

    # Build dataset
    data_config = config["data"]
    dataset = OracleTrainDataset(
        organ_reports_json=data_config["organ_reports_json"],
        oracle_contexts_json=data_config["oracle_contexts_json"],
        embeddings_npz=data_config["train_embeddings_npz"],
        embedding_keys=[data_config.get("whole_ct_key", "whole_ct_attention")] + data_config["organ_keys"],
        base_model_type=base_model_type,
        trigger_strategy=trigger_strategy,
        use_chat_template=data_config.get("use_chat_template", True),
        system_prompt=data_config.get("system_prompt", "detailed"),
        no_context_drop_rate=oracle_config.get("no_context_drop_rate", 0.1),
        whole_ct_npz=data_config.get("whole_ct_embeddings_npz_train", None),
        whole_ct_key=data_config.get("whole_ct_key", None),
        whole_ct_dim=data_config.get("whole_ct_dim", None),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=data_config.get("batch_size", 16),
        shuffle=True,
        num_workers=data_config.get("num_workers", 8),
        collate_fn=collate_fn_oracle,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    train_config = config["training"]
    opt_config = train_config["optimizer"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=opt_config["lr"],
        weight_decay=opt_config.get("weight_decay", 0.05),
        betas=tuple(opt_config.get("betas", [0.9, 0.999])),
        eps=opt_config.get("eps", 1e-8),
    )

    # C5 optimizer
    c5_optimizer = None
    if c5_head is not None:
        c5_optimizer = torch.optim.AdamW(
            c5_head.parameters(),
            lr=opt_config["lr"] * 10,  # Higher LR for small head
            weight_decay=0.01,
        )

    # Scheduler: linear warmup + cosine decay
    from torch.optim.lr_scheduler import LambdaLR
    max_steps = train_config["max_steps"]
    warmup_steps = train_config["scheduler"]["warmup_steps"]
    min_lr_ratio = train_config["scheduler"].get("min_lr_ratio", 0.01)

    def warmup_cosine_lr(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)

    # Output dir
    exp_name = config["experiment"]["name"]
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    logger.info(f"\nStarting training: {max_steps} steps")
    logger.info(f"Dataset: {len(dataset)} items")
    logger.info(f"Batch size: {data_config['batch_size']}")
    logger.info(f"Output: {output_dir}")
    save_every = train_config["save_every"]
    log_every = config["logging"]["log_every"]

    global_step = 0
    epoch = 0
    running_loss = 0.0

    while global_step < max_steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= max_steps:
                break

            should_print = (global_step % log_every == 0)

            loss, metrics = oracle_train_step(
                model=model,
                batch=batch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                step=global_step,
                config=config,
                c5_head=c5_head,
                c5_optimizer=c5_optimizer,
                ret_start_id=ret_start_id,
                ret_end_id=ret_end_id,
                should_print=should_print,
            )
            running_loss += metrics["loss"]
            global_step += 1

            # Log
            if global_step % log_every == 0:
                avg_loss = running_loss / log_every
                log_msg = f"Step {global_step}/{max_steps} | loss={avg_loss:.4f}"
                if "c5_loss" in metrics:
                    log_msg += f" | c5_loss={metrics['c5_loss']:.4f}"
                    log_msg += f" | c5_acc={metrics.get('c5_accuracy', 0):.3f}"
                if "lr" in metrics:
                    log_msg += f" | lr={metrics['lr']:.2e}"
                if "mask_ratio" in metrics:
                    log_msg += f" | mask={metrics['mask_ratio']:.2%}"
                logger.info(log_msg)
                running_loss = 0.0

            # Save
            if global_step % save_every == 0 or global_step == 1:
                ckpt_path = save_checkpoint(
                    model, tokenizer, optimizer, scheduler,
                    global_step, metrics, config, output_dir,
                    c5_head, c5_optimizer,
                )
                logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final save
    ckpt_path = save_checkpoint(
        model, tokenizer, optimizer, scheduler,
        global_step, metrics, config, output_dir,
        c5_head, c5_optimizer,
    )
    logger.info(f"Training complete! Final checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
