"""
Step 2.2: Loss masking for oracle training.

Masks tokens between <|ret_start|> and <|ret_end|> (inclusive) so the model
does not learn to generate retrieved context — only to utilize it.

For Strategy C ([RAG] token), the [RAG] token itself is NOT masked
(the model must learn to generate it).
"""

import torch
from typing import List, Tuple

from llava.constants import IGNORE_INDEX


def mask_retrieval_context_in_labels(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    ret_start_id: int,
    ret_end_id: int,
) -> torch.Tensor:
    """Mask labels for tokens between <|ret_start|> and <|ret_end|> (inclusive).

    Args:
        labels: (seq_len,) label tensor
        input_ids: (seq_len,) input ID tensor
        ret_start_id: token ID for <|ret_start|>
        ret_end_id: token ID for <|ret_end|>

    Returns:
        Modified labels with IGNORE_INDEX for masked positions
    """
    labels = labels.clone()
    seq_len = labels.shape[0]

    in_context = False
    for i in range(seq_len):
        tok = input_ids[i].item()

        if tok == ret_start_id:
            in_context = True
            labels[i] = IGNORE_INDEX  # Mask <|ret_start|> itself
        elif tok == ret_end_id and in_context:
            labels[i] = IGNORE_INDEX  # Mask <|ret_end|> itself
            in_context = False
        elif in_context:
            labels[i] = IGNORE_INDEX  # Mask everything between

    return labels


def mask_retrieval_context_batch(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    ret_start_id: int,
    ret_end_id: int,
) -> torch.Tensor:
    """Batch version of mask_retrieval_context_in_labels.

    Args:
        labels: (B, seq_len) label tensor
        input_ids: (B, seq_len) input ID tensor
        ret_start_id: token ID for <|ret_start|>
        ret_end_id: token ID for <|ret_end|>

    Returns:
        Modified labels with IGNORE_INDEX for masked positions
    """
    batch_size = labels.shape[0]
    for i in range(batch_size):
        labels[i] = mask_retrieval_context_in_labels(
            labels[i], input_ids[i], ret_start_id, ret_end_id
        )
    return labels


def count_masked_tokens(labels: torch.Tensor) -> dict:
    """Count masked vs active tokens in labels for logging.

    Args:
        labels: (B, seq_len) or (seq_len,) label tensor

    Returns:
        dict with total, masked, active, mask_ratio
    """
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    total = labels.numel()
    masked = (labels == IGNORE_INDEX).sum().item()
    active = total - masked

    return {
        "total": total,
        "masked": masked,
        "active": active,
        "mask_ratio": masked / total if total > 0 else 0,
    }
