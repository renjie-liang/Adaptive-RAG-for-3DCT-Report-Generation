"""
P2_rag model loader for inference.

Loads a PEFT checkpoint saved by P2_rag/train/train_rag_token.py.

Checkpoint folder format (produced by save_checkpoint in P2_rag/train/train.py):
    step_XXXX/
    ├── adapter_config.json        # LoRA config + _stage3_turn + _rag metadata
    ├── adapter_model.safetensors  # LoRA delta + embed_tokens + lm_head
    ├── tokenizer.json             # Full tokenizer (all special tokens pre-saved)
    ├── projector.pt               # organ_projectors weights (relative keys)
    └── training_state.pt          # step, config, metrics

Key differences vs stage3_turn/models/build_model.py:
  - No breakpoint()
  - Tokenizer is loaded directly from checkpoint (already contains ALL special tokens:
    task tokens, organ tokens, provided token, AND rag tokens <|ret_start|>/<|ret_end|>/[RAG])
    → no manual add_tokens needed during inference
  - dtype = float16  (not bfloat16; avoids NaN on long sequences during inference)
  - Reads _rag metadata from adapter_config for verification
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple

from adaragct.models.build_model import build_projectors
from adaragct.constants import ORGAN_TOKEN_NAMES


def load_model_peft(
    checkpoint_dir: str,
    device: torch.device,
    merge_lora: bool = False,
) -> Tuple[nn.Module, any]:
    """
    Load P2_rag PEFT checkpoint for inference.

    Load order:
        1. Read adapter_config.json → extract _stage3_turn (projector) + _rag (RAG tokens)
        2. Load tokenizer from checkpoint_dir (vocab already correct: base + task + organ + RAG)
        3. Load base LLaVA model
        4. resize_token_embeddings(len(tokenizer))
        5. Build projector structure
        6. Wrap with PeftModel + load adapter_model.safetensors (LoRA + embed_tokens + lm_head)
        7. Load projector.pt
        8. Optionally merge LoRA
        9. Move to device, set dtype=float16

    Args:
        checkpoint_dir: Path to PEFT checkpoint folder (e.g. .../checkpoints/step_800)
        device: Target device (cuda / cpu)
        merge_lora: If True, merge LoRA into base weights (faster inference, cannot be un-merged)

    Returns:
        (model, tokenizer)
    """
    from peft import PeftModel, LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from safetensors.torch import load_file as load_safetensors
    from llava.model.builder import load_pretrained_model

    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"[load_model_peft] Checkpoint dir not found: {ckpt_dir}")

    print(f"\n{'='*60}")
    print(f"[load_model_peft] Loading AdaRAG-CT checkpoint: {ckpt_dir}")
    print(f"{'='*60}")

    # ── 1. Read adapter_config.json ───────────────────────────────────────────
    adapter_cfg_path = ckpt_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"[load_model_peft] adapter_config.json not found: {adapter_cfg_path}")

    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)

    base_model_path = adapter_cfg["base_model_name_or_path"]

    # _stage3_turn: projector metadata
    if "_stage3_turn" not in adapter_cfg:
        raise KeyError(f"[load_model_peft] '_stage3_turn' key not found in adapter_config.json. "
                       f"Available keys: {list(adapter_cfg.keys())}")
    stage3_meta = adapter_cfg["_stage3_turn"]
    projector_type      = stage3_meta["projector_type"]
    projector_layers    = stage3_meta["projector_layers"]
    tokens_per_embedding = stage3_meta["tokens_per_embedding"]
    whole_ct_dim        = stage3_meta["whole_ct_dim"]

    # _rag: RAG token metadata (for verification only — tokenizer already has these)
    rag_meta = adapter_cfg.get("_rag", {})
    rag_tokens_in_cfg = rag_meta.get("rag_tokens", [])
    trigger_strategy   = rag_meta.get("trigger_strategy", "unknown")

    print(f"[DEBUG] base_model_path     = {base_model_path}")
    print(f"[DEBUG] projector_type      = {projector_type}")
    print(f"[DEBUG] projector_layers    = {projector_layers}")
    print(f"[DEBUG] tokens_per_embedding= {tokens_per_embedding}")
    print(f"[DEBUG] whole_ct_dim        = {whole_ct_dim}")
    print(f"[DEBUG] trigger_strategy    = {trigger_strategy}")
    print(f"[DEBUG] rag_tokens_in_cfg   = {rag_tokens_in_cfg}")

    # ── 2. Load tokenizer from checkpoint_dir ─────────────────────────────────
    # The saved tokenizer already contains ALL special tokens added during training:
    #   stage3_turn/build_model.py:  task tokens + organ tokens + provided token
    #   train_rag_token.py:          <|ret_start|>, <|ret_end|>, [RAG]
    # No manual add_tokens needed.
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=False)
    vocab_size_loaded = len(tokenizer)
    print(f"\n[DEBUG] Tokenizer loaded from checkpoint: vocab_size={vocab_size_loaded}")

    # Verify RAG tokens are present in tokenizer
    for tok in rag_tokens_in_cfg:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        unk_id = tokenizer.unk_token_id
        status = "OK" if (tok_id != unk_id and tok_id > 0) else "MISSING!"
        print(f"[DEBUG]   RAG token {tok!r:20s} → id={tok_id}  [{status}]")

    # Also verify organ tokens
    from adaragct.constants import ORGAN_TOKENS
    print(f"[DEBUG] Organ tokens check:")
    for tok in ORGAN_TOKENS:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        print(f"[DEBUG]   {tok!r:20s} → id={tok_id}")

    # ── 3. Load base LLaVA model ──────────────────────────────────────────────
    print(f"\n[DEBUG] Loading base LLaVA model: {base_model_path}")
    print(f"[DEBUG] device_map={'auto' if str(device) == 'cuda' else str(device)}")
    _, model, _, _ = load_pretrained_model(
        model_path=base_model_path,
        model_base=None,
        model_name="llava-llama",
        load_8bit=False,
        load_4bit=False,
        device_map="auto" if str(device) == "cuda" else str(device),
        device=str(device),
        use_flash_attn=False,
    )
    model.config.use_cache = False
    print(f"[DEBUG] Base model loaded: hidden_size={model.config.hidden_size}, "
          f"vocab_size={model.config.vocab_size}")

    # ── 4. Resize embeddings to match saved tokenizer ─────────────────────────
    print(f"\n[DEBUG] Resizing embeddings: {model.config.vocab_size} → {vocab_size_loaded}")
    model.resize_token_embeddings(vocab_size_loaded, mean_resizing=False)
    print(f"[DEBUG] Embeddings resized: new embed shape = "
          f"{model.get_input_embeddings().weight.shape}")

    # ── 5. Build projector structure ──────────────────────────────────────────
    hidden_size    = model.config.hidden_size
    mm_hidden_size = 256

    if whole_ct_dim is not None and whole_ct_dim != mm_hidden_size:
        mm_hidden_sizes = {
            name: (whole_ct_dim if name == "whole_ct" else mm_hidden_size)
            for name in ORGAN_TOKEN_NAMES
        }
    else:
        mm_hidden_sizes = None

    print(f"\n[DEBUG] Building projectors: type={projector_type}, layers={projector_layers}, "
          f"tokens_per_emb={tokens_per_embedding}")
    print(f"[DEBUG] mm_hidden_size={mm_hidden_size}, whole_ct_dim={whole_ct_dim}, "
          f"hidden_size={hidden_size}")
    print(f"[DEBUG] mm_hidden_sizes={mm_hidden_sizes}")

    import logging
    _dummy_logger = logging.getLogger("p2_rag.build_model")
    projector_module, _ = build_projectors(
        projector_type=projector_type,
        mm_hidden_size=mm_hidden_size,
        hidden_size=hidden_size,
        projector_layers=projector_layers,
        tokens_per_embedding=tokens_per_embedding,
        logger=_dummy_logger,
        mm_hidden_sizes=mm_hidden_sizes,
    )

    if projector_type == "shared":
        model.get_model().mm_projector = projector_module
        print(f"[DEBUG] Attached shared mm_projector")
    else:
        model.get_model().mm_projector = nn.Identity()
        model.get_model().organ_projectors = projector_module
        proj_keys = list(projector_module.keys()) if hasattr(projector_module, 'keys') else "N/A"
        print(f"[DEBUG] Attached organ_projectors (mm_projector=Identity). "
              f"Projector keys: {proj_keys}")

    # Store projector config on model.config (used by project_embeddings in inference)
    model.config.projector_type      = projector_type
    model.config.projector_layers    = projector_layers
    model.config.tokens_per_embedding = tokens_per_embedding
    model.config.mm_hidden_sizes = (
        mm_hidden_sizes if mm_hidden_sizes
        else {name: mm_hidden_size for name in ORGAN_TOKEN_NAMES}
    )
    print(f"[DEBUG] model.config.projector_type={model.config.projector_type}, "
          f"tokens_per_embedding={model.config.tokens_per_embedding}")

    # ── 6. Wrap with PeftModel + load adapter weights ─────────────────────────
    lora_config = LoraConfig(
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        target_modules=adapter_cfg["target_modules"],
        lora_dropout=adapter_cfg["lora_dropout"],
        bias=adapter_cfg["bias"],
        task_type="CAUSAL_LM",
    )
    print(f"\n[DEBUG] Building PeftModel: r={adapter_cfg['r']}, "
          f"lora_alpha={adapter_cfg['lora_alpha']}, "
          f"target_modules={adapter_cfg['target_modules']}")
    model = get_peft_model(model, lora_config)
    print(f"[DEBUG] PeftModel structure built")

    safetensors_path = ckpt_dir / "adapter_model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"[load_model_peft] adapter_model.safetensors not found: {safetensors_path}")

    print(f"[DEBUG] Loading weights from: {safetensors_path}")
    all_weights = load_safetensors(str(safetensors_path), device="cpu")
    load_result = model.load_state_dict(all_weights, strict=False)

    lora_keys   = [k for k in all_weights if "lora_" in k]
    embed_keys  = [k for k in all_weights if "embed_tokens" in k]
    lm_head_keys = [k for k in all_weights if "lm_head" in k]
    print(f"[DEBUG] Weights loaded from safetensors:")
    print(f"[DEBUG]   LoRA keys:      {len(lora_keys)}")
    print(f"[DEBUG]   embed_tokens:   {len(embed_keys)}")
    print(f"[DEBUG]   lm_head:        {len(lm_head_keys)}")
    print(f"[DEBUG]   unexpected:     {len(load_result.unexpected_keys)}")
    if load_result.unexpected_keys:
        print(f"[DEBUG]   unexpected_keys (first 3): {load_result.unexpected_keys[:3]}")

    non_lora_missing = [k for k in load_result.missing_keys if "lora_" not in k]
    if non_lora_missing:
        print(f"[DEBUG]   WARNING non-LoRA missing keys ({len(non_lora_missing)}): "
              f"{non_lora_missing[:5]}")

    # Verify embed_tokens shape matches tokenizer vocab
    try:
        base = model.get_model() if hasattr(model, 'get_model') else model.model
        embed_shape = model.get_input_embeddings().weight.shape
        print(f"[DEBUG] embed_tokens shape after load: {embed_shape} "
              f"(expected [{vocab_size_loaded}, {hidden_size}])")
        assert embed_shape[0] == vocab_size_loaded, (
            f"[ASSERT] embed_tokens size mismatch: "
            f"got {embed_shape[0]}, expected {vocab_size_loaded}"
        )
    except Exception as e:
        print(f"[DEBUG] WARNING: embed shape check failed: {e}")

    # ── 7. Optionally merge LoRA ──────────────────────────────────────────────
    if merge_lora:
        model = model.merge_and_unload()
        print(f"[DEBUG] LoRA merged and unloaded")

    # ── 8. Load projector weights ─────────────────────────────────────────────
    projector_pt = ckpt_dir / "projector.pt"
    if projector_pt.exists():
        proj_state = torch.load(projector_pt, map_location="cpu", weights_only=False)
        print(f"\n[DEBUG] Loading projector.pt: {len(proj_state)} keys")
        print(f"[DEBUG] projector.pt keys (first 5): {list(proj_state.keys())[:5]}")

        if projector_type == "shared":
            load_r = model.get_model().mm_projector.load_state_dict(proj_state, strict=True)
        else:
            load_r = model.get_model().organ_projectors.load_state_dict(proj_state, strict=True)

        print(f"[DEBUG] Projector weights loaded successfully ({len(proj_state)} keys)")

        # Verify projector weights are not random (sanity check)
        try:
            base = model.get_model() if hasattr(model, 'get_model') else model.model
            if hasattr(base, 'organ_projectors'):
                sample_key = next(iter(base.organ_projectors.keys()))
                sample_w = next(base.organ_projectors[sample_key].parameters())
                print(f"[DEBUG] organ_projectors[{sample_key!r}] weight: "
                      f"shape={sample_w.shape}, "
                      f"mean={sample_w.float().mean().item():.6f}, "
                      f"std={sample_w.float().std().item():.6f}")
            elif hasattr(base, 'mm_projector') and not isinstance(base.mm_projector, nn.Identity):
                sample_w = next(base.mm_projector.parameters())
                print(f"[DEBUG] mm_projector weight: shape={sample_w.shape}, "
                      f"mean={sample_w.float().mean().item():.6f}, "
                      f"std={sample_w.float().std().item():.6f}")
        except Exception as e:
            print(f"[DEBUG] WARNING: projector weight check failed: {e}")
    else:
        print(f"[DEBUG] WARNING: projector.pt NOT FOUND in {ckpt_dir} "
              f"— projector weights are RANDOM!")

    # ── 9. Set dtype=float16 + move projectors to device ──────────────────────
    # float16 is required for inference (bfloat16 causes NaN on long sequences without KV cache)
    print(f"\n[DEBUG] Converting model to float16 and moving projectors to {device}")
    model.to(dtype=torch.float16)
    model.get_model().mm_projector.to(device=device)
    if hasattr(model.get_model(), "organ_projectors"):
        model.get_model().organ_projectors.to(device=device)

    total_params   = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[DEBUG] Total params: {total_params:.1f}M, "
          f"trainable: {trainable_params:.1f}M")
    print(f"[DEBUG] dtype=float16, projectors on device={device}")

    print(f"\n{'='*60}")
    print(f"[load_model_peft] Model loaded successfully")
    print(f"  projector_type      = {model.config.projector_type}")
    print(f"  tokens_per_embedding= {model.config.tokens_per_embedding}")
    print(f"  tokenizer vocab_size= {len(tokenizer)}")
    print(f"  dtype               = float16")
    print(f"  device              = {device}")
    print(f"{'='*60}\n")

    return model, tokenizer
