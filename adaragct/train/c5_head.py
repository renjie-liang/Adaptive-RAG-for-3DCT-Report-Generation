"""
Step 2.3: C5 Head — Learned retrieval necessity predictor.

Strategy B only. A linear classifier on the LLM's hidden state at sentence
boundary positions. Predicts whether RAG is needed at each boundary.

Training labels come for free from the oracle training data (needs_rag field).
Trained with BCE loss as auxiliary loss: total_loss = lm_loss + alpha * c5_loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class C5Head(nn.Module):
    """Binary classifier on LLM hidden states at sentence boundaries.

    Predicts P(needs_rag) at each sentence boundary position.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_positions: List[List[int]],
        boundary_labels: Optional[List[List[int]]] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict]:
        """
        Forward pass for C5 head.

        Args:
            hidden_states: (B, seq_len, hidden_size) from LLM
            boundary_positions: list of lists, each inner list contains
                token positions where sentences end (one per batch item)
            boundary_labels: list of lists, each inner list contains
                binary labels (0/1) for needs_rag at each boundary.
                If None, returns logits only (inference mode).

        Returns:
            loss: scalar BCE loss if boundary_labels provided, else None
            info: dict with logits, predictions, accuracy
        """
        device = hidden_states.device
        all_logits = []
        all_labels = []

        for b in range(hidden_states.shape[0]):
            positions = boundary_positions[b]
            if not positions:
                continue

            # Clamp positions to valid range
            valid_positions = [p for p in positions if 0 <= p < hidden_states.shape[1]]
            if not valid_positions:
                continue

            pos_tensor = torch.tensor(valid_positions, dtype=torch.long, device=device)
            boundary_hidden = hidden_states[b, pos_tensor]  # (num_boundaries, hidden_size)
            logits = self.classifier(boundary_hidden).squeeze(-1)  # (num_boundaries,)
            all_logits.append(logits)

            if boundary_labels is not None:
                labels = boundary_labels[b]
                # Align labels with valid positions
                valid_labels = [labels[i] for i, p in enumerate(positions) if 0 <= p < hidden_states.shape[1]]
                label_tensor = torch.tensor(valid_labels, dtype=torch.float, device=device)
                all_labels.append(label_tensor)

        info = {}

        if not all_logits:
            info["n_boundaries"] = 0
            return None, info

        all_logits_cat = torch.cat(all_logits)
        info["n_boundaries"] = len(all_logits_cat)
        info["logits"] = all_logits_cat.detach()

        # Predictions
        preds = (torch.sigmoid(all_logits_cat) > 0.5).long()
        info["predictions"] = preds.detach()

        if boundary_labels is not None and all_labels:
            all_labels_cat = torch.cat(all_labels)
            loss = F.binary_cross_entropy_with_logits(all_logits_cat, all_labels_cat)

            # Accuracy
            accuracy = (preds == all_labels_cat.long()).float().mean().item()
            info["accuracy"] = accuracy
            info["loss"] = loss.item()
            info["n_positive"] = all_labels_cat.sum().item()
            info["n_negative"] = len(all_labels_cat) - info["n_positive"]

            return loss, info

        return None, info


def find_sentence_boundary_positions(
    input_ids: torch.Tensor,
    tokenizer,
    prompt_len: int,
    ret_start_id: Optional[int] = None,
) -> List[int]:
    """Find token positions that correspond to sentence boundaries in the target.

    Primary: positions just before <|ret_start|> tokens (these are the exact
    positions where oracle context was injected, matching c5_labels).
    Fallback: period tokens followed by a space token (for sentences without
    context injection).

    Args:
        input_ids: (seq_len,) token IDs
        tokenizer: tokenizer for decoding
        prompt_len: length of prompt (skip these tokens)
        ret_start_id: token ID for <|ret_start|> (if available)

    Returns:
        List of token positions (absolute, in input_ids) at sentence boundaries
    """
    ids_list = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    positions = []

    # Use <|ret_start|> positions as ground-truth sentence boundaries
    if ret_start_id is not None:
        for i in range(prompt_len, len(ids_list)):
            if ids_list[i] == ret_start_id and i > prompt_len:
                positions.append(i - 1)  # Position just before context injection

    # If no ret_start found, fall back to period-based detection
    if not positions:
        period_ids = tokenizer.encode(".", add_special_tokens=False)
        if period_ids:
            period_id = period_ids[-1]
            space_id = tokenizer.encode(" ", add_special_tokens=False)
            space_id = space_id[0] if space_id else None
            for i in range(prompt_len, len(ids_list) - 1):
                if ids_list[i] == period_id:
                    # Only count as boundary if followed by space (skip "2.5", "e.g.")
                    if space_id is not None and ids_list[i + 1] == space_id:
                        positions.append(i)

    return positions
