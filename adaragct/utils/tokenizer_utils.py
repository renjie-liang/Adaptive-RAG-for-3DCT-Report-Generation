"""
Tokenizer utilities for handling organ tokens

Similar to llava.mm_utils.tokenizer_image_token but for organ-specific tokens.
"""
import torch
from typing import List, Optional, Union

from adaragct.constants import (
    ORGAN_TOKENS,
    ORGAN_TOKEN_TO_INDEX,
    ORGAN_TOKEN_INDICES,
)


def tokenizer_organ_token(
    prompt: str,
    tokenizer,
    return_tensors: Optional[str] = None,
    add_special_tokens: bool = True,  # 1. 增加这个参数
) -> Union[List[int], torch.Tensor]:
    """
    Tokenize a prompt containing organ tokens (<whole_ct>, <lung>, etc.).

    Organ tokens are replaced with their special negative indices (similar to IMAGE_TOKEN_INDEX).
    These indices will later be replaced with actual visual embeddings during model forward pass.

    Args:
        prompt: Input prompt string containing organ tokens
        tokenizer: HuggingFace tokenizer
        return_tensors: If "pt", return PyTorch tensor; otherwise return list

    Returns:
        Token IDs with organ tokens replaced by their special indices
    """
    # Split prompt by organ tokens
    prompt_chunks = [prompt]

    for organ_token in ORGAN_TOKENS:
        new_chunks = []
        for chunk in prompt_chunks:
            if isinstance(chunk, str):
                parts = chunk.split(organ_token)
                for i, part in enumerate(parts):
                    if part:
                        new_chunks.append(part)
                    if i < len(parts) - 1:
                        new_chunks.append(organ_token)
            else:
                new_chunks.append(chunk)
        prompt_chunks = new_chunks

    # 修改最后的分词循环部分
    input_ids = []
    for chunk in prompt_chunks:
        if chunk in ORGAN_TOKEN_TO_INDEX:
            input_ids.append(ORGAN_TOKEN_TO_INDEX[chunk])
        elif isinstance(chunk, str) and chunk:
            # 2. 这里传递 add_special_tokens
            # 注意：因为我们是分块分词，只有第一个有效 chunk 应该考虑这个参数
            # 但在 SFT 拼接场景下，target 部分通常显式设为 False
            chunk_ids = tokenizer(chunk, add_special_tokens=add_special_tokens).input_ids
            input_ids.extend(chunk_ids)
            
            # 关键：一旦处理完第一个 chunk，后续 chunk 不应再加特殊 token (如 <s>)
            # 否则序列中间会出现多个起始符
            add_special_tokens = False


    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def get_organ_token_mask(input_ids: Union[List[int], torch.Tensor]) -> torch.Tensor:
    """
    Create a boolean mask indicating positions of organ tokens.

    Args:
        input_ids: Token IDs (list or tensor)

    Returns:
        Boolean tensor where True indicates organ token positions
    """
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids)

    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for idx in ORGAN_TOKEN_INDICES:
        mask |= (input_ids == idx)

    return mask


def count_organ_tokens(input_ids: Union[List[int], torch.Tensor]) -> int:
    """
    Count the number of organ tokens in input_ids.

    Args:
        input_ids: Token IDs (list or tensor)

    Returns:
        Number of organ tokens
    """
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()

    return sum(1 for tok_id in input_ids if tok_id in ORGAN_TOKEN_INDICES)
