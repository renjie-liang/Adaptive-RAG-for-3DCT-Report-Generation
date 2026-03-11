"""
Training step for LLaVA report generation with multiple embeddings

Handles 5 embeddings: whole_ct_attention + 4 organs (lung, heart, esophagus, aorta)
Fail-fast principles: No try-except, explicit errors
"""
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple
from torch.amp import autocast, GradScaler

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token

from adaragct.data.dataset import trunc_str
from adaragct.constants import (
    ORGAN_TOKENS,
    ORGAN_TOKEN_TO_INDEX,
    ORGAN_INDEX_TO_SLOT,
    ORGAN_TOKEN_INDICES,
    INDEX_TO_ORGAN_TOKEN,
    NUM_ORGAN_TOKENS,
)
from adaragct.utils.tokenizer_utils import tokenizer_organ_token


def tokenizer_with_organ_tokens(
    prompt: str,
    tokenizer,
    return_tensors: str = None,
) -> List[int]:
    """
    Tokenize prompt with organ tokens, converting them to IMAGE_TOKEN_INDEX.

    This maintains compatibility with LLaVA's multimodal processing while
    using semantic organ tokens in the prompt.

    Organ tokens (<whole_ct>, <lung>, etc.) are converted to IMAGE_TOKEN_INDEX (-200)
    so LLaVA's prepare_inputs_labels_for_multimodal can replace them with embeddings.
    """
    # Replace organ tokens with a placeholder that won't be split
    # We use DEFAULT_IMAGE_TOKEN which tokenizer_image_token handles
    modified_prompt = prompt
    for organ_token in ORGAN_TOKENS:
        modified_prompt = modified_prompt.replace(organ_token, DEFAULT_IMAGE_TOKEN)

    # Use LLaVA's tokenizer_image_token to handle IMAGE_TOKEN_INDEX conversion
    return tokenizer_image_token(
        prompt=modified_prompt,
        tokenizer=tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors=return_tensors,
    )


def project_embeddings(
    embeddings: torch.Tensor,
    model: nn.Module,
    has_images: list,
    device: torch.device,
) -> list:
    """Project embeddings through the appropriate projector(s).
    
    Handles shared, independent, and grouped projector types,
    as well as multi-token output (tokens_per_embedding > 1).
    
    Args:
        embeddings: (B, num_embeddings, 256) raw ViSD-Boost embeddings
        model: LLaVA model (may be PEFT-wrapped)
        has_images: list of bools indicating which samples have images
        device: target device
    
    Returns:
        images_list: list of projected embedding tensors, ready for model forward.
                     For shared projector: each is (1, 1, 1, 1, 256) (raw, model projects internally)
                     For independent/grouped: each is (1, tokens_per_embedding, hidden_size) (pre-projected)
    """
    # Get the base model (unwrap PEFT if needed)
    base_model = model.model if hasattr(model, 'model') else model
    if hasattr(base_model, 'model'):
        inner_model = base_model.model
    else:
        inner_model = base_model
    
    projector_type = getattr(model.config, 'projector_type', 'shared')
    tokens_per_embedding = getattr(model.config, 'tokens_per_embedding', 1)
    hidden_size = model.config.hidden_size
    mm_hidden_sizes = getattr(model.config, 'mm_hidden_sizes', None)
    batch_size = embeddings.shape[0]
    num_embeddings = embeddings.shape[1]
    
    from adaragct.constants import ORGAN_TOKEN_NAMES
    
    images_list = []
    
    if projector_type == "shared":
        # Shared projector: pass raw embeddings, LLaVA's encode_images handles projection
        for i in range(batch_size):
            if has_images[i]:
                sample_embeddings = embeddings[i].to(device=device, dtype=torch.bfloat16)
                for j in range(num_embeddings):
                    organ_name = ORGAN_TOKEN_NAMES[j]
                    input_dim = mm_hidden_sizes[organ_name] if mm_hidden_sizes else sample_embeddings.shape[-1]
                    emb_j = sample_embeddings[j, :input_dim]
                    if tokens_per_embedding == 1:
                        images_list.append(emb_j.view(1, 1, 1, 1, -1))
                    else:
                        # Multi-token: project here, output (1, N, hidden_size)
                        emb = emb_j.unsqueeze(0)  # (1, input_dim)
                        proj_out = inner_model.mm_projector(emb)  # (1, N*hidden_size)
                        proj_out = proj_out.view(1, tokens_per_embedding, hidden_size)  # (1, N, H)
                        images_list.append(proj_out)
    else:
        # Independent or grouped: pre-project with organ-specific projectors
        organ_projectors = inner_model.organ_projectors
        proj_dtype = next(organ_projectors.parameters()).dtype
        for i in range(batch_size):
            if has_images[i]:
                sample_embeddings = embeddings[i].to(device=device, dtype=proj_dtype)
                for j in range(num_embeddings):
                    organ_name = ORGAN_TOKEN_NAMES[j]
                    input_dim = mm_hidden_sizes[organ_name] if mm_hidden_sizes else sample_embeddings.shape[-1]
                    emb = sample_embeddings[j, :input_dim].unsqueeze(0)  # (1, input_dim)
                    
                    if projector_type == "independent":
                        proj = organ_projectors[organ_name]
                    elif projector_type == "grouped":
                        proj = organ_projectors["global"] if organ_name == "whole_ct" else organ_projectors["organ"]
                    else:
                        raise ValueError(f"Unknown projector_type: {projector_type}")
                    
                    proj_out = proj(emb)  # (1, hidden_size) or (1, N*hidden_size)
                    if tokens_per_embedding > 1:
                        # Output (N, H): LLaVA treats each row as one image token
                        proj_out = proj_out.view(tokens_per_embedding, hidden_size)
                    else:
                        proj_out = proj_out.view(1, hidden_size)
                    images_list.append(proj_out)
    
    return images_list


def train_step(
    model: nn.Module,
    batch: Dict[str, Any],
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    scaler: GradScaler,
    step: int,
    grad_accum_steps: int = 1,
    use_amp: bool = True,
    prompt_template: str = None,
    max_seq_length_by_task: Dict[str, int] = None,
    should_print: bool = False,
    label_smoothing: float = 0.0,
    use_chat_template: bool = False,
) -> Tuple[float, Dict[str, float]]:
    """
    Single training step for LLaVA (report-only or VQA joint).

    Batch may be:
    - VQA: 'embeddings', 'prompt' (list), 'target' (list), 'task_type', ...
    - Legacy: 'embeddings', 'report' (list), 'study_id' (list); use prompt_template.
    """
    model.train()

    embeddings = batch["embeddings"].to(device)  # (B, num_embeddings, 256)
    batch_size = embeddings.shape[0]
    num_embeddings = embeddings.shape[1]

    if "prompt" in batch and "target" in batch:
        prompts = batch["prompt"]
        targets = batch["target"]
    else:
        raise KeyError(
            "Batch must have either ('prompt', 'target') for VQA or 'report' for legacy; "
            f"got keys: {list(batch.keys())}"
        )
    task_types = batch["task_type"]
    full_ids_list = []
    labels_list = []
    has_images = batch["has_image"]
    # Get EOS token for proper sequence termination
    eos_token = tokenizer.eos_token
    if eos_token is None:
        raise ValueError("tokenizer.eos_token is None; EOS token is required for training")

    for i in range(batch_size):
        prompt_text = prompts[i]
        target_text = targets[i]
        task = task_types[i]
        max_seq_length = max_seq_length_by_task[task]
        has_image = has_images[i]
        # Prompts now come with organ tokens already included from dataset
        # For image samples: "<whole_ct>\n<lung>\n<heart>\n<esophagus>\n<aorta>\n<task_token>\nquestion"
        # For paper samples: "<provided>\n<task_token>\nquestion"
        prompt_with_tokens = prompt_text

        # Add EOS token to target_text for proper sequence termination
        # When using chat template, target already ends with <|eot_id|>, so use that as EOS
        if use_chat_template:
            target_with_eos = target_text  # already has <|eot_id|> from dataset
        else:
            target_with_eos = target_text + eos_token
        full_text = prompt_with_tokens + target_with_eos

        # --- 修改后的逻辑 ---

        # 1. 分别对 prompt 和 target 进行分词
        prompt_ids = tokenizer_organ_token(
            prompt=prompt_with_tokens,
            tokenizer=tokenizer,
            return_tensors=None,
            add_special_tokens=True  # 这里保留 BOS
        )

        target_ids = tokenizer_organ_token(
            prompt=target_with_eos,
            tokenizer=tokenizer,
            return_tensors=None,
            add_special_tokens=False # 关键：防止 target 前面被自动加上 <s> 等起始符
        )

        if len(target_ids) > 0 and target_ids[0] == tokenizer.bos_token_id:
            target_ids = target_ids[1:] # 强制切掉多余的起始符

        # 2. 手动拼接 IDs，这样可以百分之百保证长度对齐
        full_ids = prompt_ids + target_ids
        prompt_len = len(prompt_ids)

        # 3. 构造 labels
        labels = [IGNORE_INDEX] * prompt_len + target_ids

        # 4. 之后再进行截断处理
        if len(full_ids) > max_seq_length:
            full_ids = full_ids[:max_seq_length]
            labels = labels[:max_seq_length]

        full_ids_list.append(full_ids)
        labels_list.append(labels)



    max_seq_len = max(len(ids) for ids in full_ids_list)
    pad_token_id = tokenizer.pad_token_id 
    assert pad_token_id is not None, "tokenizer.pad_token_id is None"
    full_ids_padded, labels_padded, attention_mask = [], [], []
    for full_ids, labels in zip(full_ids_list, labels_list):
        seq_len = len(full_ids)
        padding_len = max_seq_len - seq_len
        full_ids_padded.append(full_ids + [pad_token_id] * padding_len)
        labels_padded.append(labels + [IGNORE_INDEX] * padding_len)
        attention_mask.append([1] * seq_len + [0] * padding_len)
    
    # Convert to tensors
    full_ids = torch.tensor(full_ids_padded, dtype=torch.long).to(device)
    labels = torch.tensor(labels_padded, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
    
    # 2. 混合精度前向传播 (关键修改)
    # 强制使用 bfloat16，这是 B200 的"母语"
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
        # Project embeddings through appropriate projector(s)
        images_list = project_embeddings(
            embeddings=embeddings,
            model=model,
            has_images=batch["has_image"],
            device=device,
        )
        
        outputs = model(
            input_ids=full_ids,
            attention_mask=attention_mask,
            labels=labels,
            images=images_list,
        )
        
        # Apply label smoothing if configured
        if label_smoothing > 0.0:
            # Recompute loss with label smoothing
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(
                ignore_index=IGNORE_INDEX, 
                label_smoothing=label_smoothing
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ) / grad_accum_steps
        else:
            loss = outputs.loss / grad_accum_steps

    # 3. 反向传播 (彻底删掉 scaler.scale)
    # 因为 BF16 几乎不会发生梯度下溢，直接 backward 即可
    loss.backward()

    metrics = {'loss': loss.item() * grad_accum_steps}
    is_accum_step = (step + 1) % grad_accum_steps == 0
    
    # 4. 优化器更新 (彻底删掉 scaler.unscale_ 和 scaler.update)
    if is_accum_step:
        # 直接进行梯度裁剪，不再需要 unscale_
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        metrics['grad_norm'] = grad_norm.item()
        
        # 直接使用 optimizer.step()
        optimizer.step()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
            metrics['lr'] = scheduler.get_last_lr()[0]
    
    return loss.item() * grad_accum_steps, metrics