"""
Build LLaVA model with LoRA for report generation

Uses ViSD-Boost embeddings (5 embeddings: whole_ct + 4 organs) as input.
Fail-fast principles: No try-except, direct config access, explicit errors.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from pathlib import Path

from adaragct.utils.logger import get_module_logger
from llava.model.builder import load_pretrained_model
from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    TOKEN_FOR_REPORT_GENERATION,
    IMAGE_TOKEN_INDEX,
)
from llava.train.train import smart_tokenizer_and_embedding_resize

from adaragct.constants import ORGAN_TOKENS, ORGAN_TOKEN_NAMES, PROVIDED_TOKEN


def _build_mlp(input_size: int, hidden_size: int, num_layers: int = 2) -> nn.Sequential:
    """Build an MLP projector with configurable depth."""
    if num_layers == 2:
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
    elif num_layers == 3:
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
    else:
        raise ValueError(f"Unsupported projector_layers={num_layers}, must be 2 or 3")


def _build_multi_token_mlp(input_size: int, hidden_size: int, tokens_per_embedding: int, num_layers: int = 2) -> nn.Sequential:
    """Build an MLP projector that outputs multiple tokens per embedding.
    
    Maps input_size -> tokens_per_embedding * hidden_size, then reshape.
    """
    output_size = tokens_per_embedding * hidden_size
    if num_layers == 2:
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
    elif num_layers == 3:
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
    else:
        raise ValueError(f"Unsupported projector_layers={num_layers}, must be 2 or 3")


def build_projectors(
    projector_type: str,
    mm_hidden_size: int,
    hidden_size: int,
    projector_layers: int,
    tokens_per_embedding: int,
    logger,
    mm_hidden_sizes: Dict[str, int] = None,
) -> Tuple[nn.Module, dict]:
    """Build projector module(s) based on config.
    
    Args:
        mm_hidden_size: default input dim for all projectors (used if mm_hidden_sizes is None)
        mm_hidden_sizes: optional per-organ input dims, e.g. {"whole_ct": 512, "lung": 256, ...}
    
    Returns:
        projector_module: nn.Module to attach to model (nn.Sequential or nn.ModuleDict)
        projector_info: dict with metadata about the projector setup
    """
    info = {
        "type": projector_type,
        "layers": projector_layers,
        "tokens_per_embedding": tokens_per_embedding,
        "mm_hidden_size": mm_hidden_size,
        "mm_hidden_sizes": mm_hidden_sizes,
        "hidden_size": hidden_size,
    }
    
    is_multi_token = tokens_per_embedding > 1

    def _call_build_fn(input_dim):
        if is_multi_token:
            return _build_multi_token_mlp(input_dim, hidden_size, tokens_per_embedding, num_layers=projector_layers)
        else:
            return _build_mlp(input_dim, hidden_size, projector_layers)

    if projector_type == "shared":
        if mm_hidden_sizes and len(set(mm_hidden_sizes.values())) > 1:
            raise ValueError(f"shared projector requires uniform input dims, got {mm_hidden_sizes}")
        proj = _call_build_fn(mm_hidden_size)
        logger.info(f"🔧 Projector: shared, {projector_layers}-layer MLP, tokens_per_embedding={tokens_per_embedding}")
        return proj, info
    
    elif projector_type == "independent":
        projectors = nn.ModuleDict()
        for name in ORGAN_TOKEN_NAMES:
            input_dim = mm_hidden_sizes[name] if mm_hidden_sizes else mm_hidden_size
            projectors[name] = _call_build_fn(input_dim)
        if mm_hidden_sizes:
            logger.info(f"🔧 Projector: independent (5 MLPs), {projector_layers}-layer, per-organ dims={mm_hidden_sizes}, tokens_per_embedding={tokens_per_embedding}")
        else:
            logger.info(f"🔧 Projector: independent (5 MLPs), {projector_layers}-layer, tokens_per_embedding={tokens_per_embedding}")
        return projectors, info
    
    elif projector_type == "grouped":
        if mm_hidden_sizes and len(set(mm_hidden_sizes.values())) > 1:
            raise ValueError(f"grouped projector requires uniform input dims, got {mm_hidden_sizes}")
        projectors = nn.ModuleDict({
            "global": _call_build_fn(mm_hidden_size),
            "organ": _call_build_fn(mm_hidden_size),
        })
        logger.info(f"🔧 Projector: grouped (global + organ), {projector_layers}-layer, tokens_per_embedding={tokens_per_embedding}")
        return projectors, info
    
    else:
        raise ValueError(f"Unknown projector_type='{projector_type}', must be 'shared', 'independent', or 'grouped'")


def find_all_linear_names(model):
    """Find all linear layer names for LoRA targeting"""
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    # Filter out pure digit names to avoid matching model.layers.0
    lora_module_names = {name for name in lora_module_names if not name.isdigit()}
    
    return list(lora_module_names)

def build_model(
    config: Dict,
    device: torch.device,
    load_8bit: bool = False,
    load_4bit: bool = False,
    train_projector_only: bool = False,
    pretrain_checkpoint: str = None,
) -> Tuple[nn.Module, any]:
    logger = get_module_logger()
    model_config = config['model']
    
    # 1. 基础参数加载
    model_path = model_config['model_path']
    model_name = model_config['model_name']
    use_flash_attn = model_config['use_flash_attn']
    
    # 打印基础模型信息
    logger.info(f"🤖 Loading base model: {model_name}")
    logger.info(f"📂 Model path: {model_path}")
    logger.info(f"⚡ Flash attention: {use_flash_attn}")
    logger.info(f"💾 Quantization: 8bit={load_8bit}, 4bit={load_4bit}")
    
    # 2. 加载模型（保持原始加载精度，通常是 FP16 或 BF16）
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        device_map="auto" if str(device) == "cuda" else str(device),
        device=str(device),
        use_flash_attn=use_flash_attn,
    )
    model.config.use_cache = False 
    logger.info(f"✅ Base model loaded successfully")
    logger.info(f"📊 Model config: hidden_size={model.config.hidden_size}, vocab_size={model.config.vocab_size}")
    logger.info(f"🔤 Tokenizer vocab size: {len(tokenizer)}") 

    
    # 3. Projector 初始化/调整
    hidden_size = model.config.hidden_size
    mm_hidden_size = 256 
    
    # Read projector config with defaults for backward compatibility
    projector_type = model_config['projector_type']
    projector_layers = model_config['projector_layers']
    tokens_per_embedding = model_config['tokens_per_embedding']
    
    # Build per-organ input dim mapping (for heterogeneous embedding dims like CT-CLIP)
    whole_ct_dim = config['data']['whole_ct_dim']
    if whole_ct_dim is not None and whole_ct_dim != mm_hidden_size:
        mm_hidden_sizes = {name: (whole_ct_dim if name == 'whole_ct' else mm_hidden_size)
                          for name in ORGAN_TOKEN_NAMES}
        logger.info(f"🔧 Per-organ input dims: {mm_hidden_sizes}")
    else:
        mm_hidden_sizes = None
    
    # Build projector(s) based on config
    projector_module, projector_info = build_projectors(
        projector_type=projector_type,
        mm_hidden_size=mm_hidden_size,
        hidden_size=hidden_size,
        projector_layers=projector_layers,
        tokens_per_embedding=tokens_per_embedding,
        logger=logger,
        mm_hidden_sizes=mm_hidden_sizes,
    )
    
    # Attach projector(s) to model
    if projector_type == "shared":
        # For shared projector, use mm_projector (LLaVA standard)
        model.get_model().mm_projector = projector_module
    else:
        # For independent/grouped, organ_projectors handle projection in train_step.
        # mm_projector is set to Identity so LLaVA's encode_images is a pass-through.
        model.get_model().mm_projector = nn.Identity()
        model.get_model().organ_projectors = projector_module
    
    # Store projector config on model for downstream use
    model.config.projector_type = projector_type
    model.config.projector_layers = projector_layers
    model.config.tokens_per_embedding = tokens_per_embedding
    model.config.mm_hidden_sizes = mm_hidden_sizes if mm_hidden_sizes else {name: mm_hidden_size for name in ORGAN_TOKEN_NAMES}
    
    logger.info(f"🔧 Projector initialized: {mm_hidden_size} -> {hidden_size}")

    # 3b. 加载预训练checkpoint中的 projector 权重（PEFT wrap 之前）
    # 必须在 PEFT wrap 之前加载，否则 key name 会因为 PEFT 前缀而不匹配
    _ckpt_state_dict = None
    if pretrain_checkpoint and Path(pretrain_checkpoint).exists():
        projector_keywords = ['mm_projector', 'organ_projectors']
        ckpt_path = Path(pretrain_checkpoint)

        if ckpt_path.is_dir():
            # PEFT folder format detected — build_model() only loads projector,
            # which silently drops LoRA/embed_tokens/lm_head.
            # Use build_model_from_peft() instead to load ALL weights correctly.
            assert not ((ckpt_path / "adapter_config.json").exists() and (ckpt_path / "adapter_model.safetensors").exists()), (
                f"pretrain_checkpoint is a full PEFT folder: {ckpt_path}\n"
                f"build_model() cannot correctly load PEFT checkpoints "
                f"(it only loads projector, dropping LoRA/embed_tokens/lm_head).\n"
                f"Use build_model_from_peft() instead."
            )
            # Merged model directory (from merge_peft_checkpoint.py):
            # has projector.pt but no adapter_config.json
            projector_pt = ckpt_path / "projector.pt"
            if projector_pt.exists():
                proj_state = torch.load(projector_pt, map_location="cpu", weights_only=False)
                if projector_type == "shared":
                    model.get_model().mm_projector.load_state_dict(proj_state, strict=True)
                else:
                    model.get_model().organ_projectors.load_state_dict(proj_state, strict=True)
                logger.info(f"📦 Loaded projector.pt from merged dir: {len(proj_state)} keys")
            else:
                logger.warning(f"⚠️ Directory {ckpt_path} has no adapter_config.json and no projector.pt")
        else:
            # ── Old .pt format ──────────────────────────────────────────────
            _ckpt = torch.load(pretrain_checkpoint, map_location="cpu", weights_only=False)
            _ckpt_state_dict = _ckpt['model_state_dict']
            _ckpt_step = _ckpt['step']

            # 提取 projector 权重并加载 (handles both mm_projector and organ_projectors)
            projector_keys_raw = {k: v for k, v in _ckpt_state_dict.items()
                                  if any(kw in k for kw in projector_keywords)}
            # Strip PEFT prefix 'base_model.model.' so keys match pre-PEFT model structure
            # Phase 1 keys: 'model.organ_projectors.*'
            # Phase 2 keys: 'base_model.model.model.organ_projectors.*' → strip to 'model.organ_projectors.*'
            projector_keys = {}
            for k, v in projector_keys_raw.items():
                stripped = k
                if k.startswith("base_model.model."):
                    stripped = k[len("base_model.model."):]
                projector_keys[stripped] = v
            if projector_keys:
                load_result = model.load_state_dict(projector_keys, strict=False)
                loaded_count = len(projector_keys) - len([k for k in load_result.missing_keys
                                                           if any(kw in k for kw in projector_keywords)])
                logger.info(f"📦 Pre-PEFT: Loaded {loaded_count}/{len(projector_keys)} projector keys from step {_ckpt_step}")
            else:
                # Fallback: load projector from original pretrain_checkpoint in config
                # This handles Phase 2 checkpoints that only saved LoRA weights (freeze_projector=True)
                fallback_ckpt = config['model']['pretrain_checkpoint']
                if fallback_ckpt and Path(fallback_ckpt).exists() and str(Path(fallback_ckpt).resolve()) != str(Path(pretrain_checkpoint).resolve()):
                    logger.info(f"📦 Pre-PEFT: No projector keys in checkpoint (step {_ckpt_step}), trying fallback: {fallback_ckpt}")
                    _fb_ckpt = torch.load(fallback_ckpt, map_location="cpu", weights_only=False)
                    _fb_state = _fb_ckpt['model_state_dict']
                    fb_proj_keys = {k: v for k, v in _fb_state.items()
                                   if any(kw in k for kw in projector_keywords)}
                    if fb_proj_keys:
                        load_result = model.load_state_dict(fb_proj_keys, strict=False)
                        loaded_count = len(fb_proj_keys) - len([k for k in load_result.missing_keys
                                                                if any(kw in k for kw in projector_keywords)])
                        logger.info(f"📦 Pre-PEFT: Loaded {loaded_count}/{len(fb_proj_keys)} projector keys from fallback checkpoint")
                    else:
                        logger.warning(f"⚠️ Pre-PEFT: No projector keys in fallback checkpoint either!")
                    del _fb_ckpt, _fb_state
                else:
                    logger.warning(f"⚠️ Pre-PEFT: No projector keys in checkpoint (step {_ckpt_step}) and no fallback available")
    elif pretrain_checkpoint:
        raise FileNotFoundError(f"Pretrain checkpoint not found: {pretrain_checkpoint}")

    # 4. Token 处理（保持不变，这是业务逻辑）
    from llava.constants import TOKEN_FOR_MULTIPLE_CHOICE, TOKEN_FOR_LONG_ANSWER, TOKEN_FOR_SHORT_ANSWER
    task_tokens = [TOKEN_FOR_MULTIPLE_CHOICE, TOKEN_FOR_LONG_ANSWER, TOKEN_FOR_SHORT_ANSWER, TOKEN_FOR_REPORT_GENERATION]
    tokenizer.add_tokens(task_tokens, special_tokens=True)
    tokenizer.add_tokens(ORGAN_TOKENS, special_tokens=True)
    tokenizer.add_tokens([PROVIDED_TOKEN], special_tokens=True)
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token="<pad>"))
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    
    logger.info(f"🎯 Added {len(task_tokens)} task tokens + {len(ORGAN_TOKENS)} organ tokens + 1 provided token")
    logger.info(f"🔤 Final tokenizer vocab size: {len(tokenizer)}")

    # 5. 训练模式设置
    if train_projector_only:
        # 模式 A: 只有 Projector 可训练
        model.requires_grad_(False)
        for param in model.get_model().mm_projector.parameters():
            param.requires_grad = True
        # Also unfreeze organ_projectors if they exist (independent/grouped)
        if hasattr(model.get_model(), 'organ_projectors'):
            for param in model.get_model().organ_projectors.parameters():
                param.requires_grad = True
        logger.info(f"🎯 Training mode: Projector ONLY (all other layers frozen)")
    else:
        # 模式 B: LoRA 微调
        from peft import LoraConfig, get_peft_model
        
        # Read target_modules from config, fallback to auto-detect
        target_modules = model_config['lora_target_modules']
        if target_modules is None:
            target_modules = find_all_linear_names(model)
            logger.info(f"🔍 Auto-detected target modules: {target_modules}")
        else:
            logger.info(f"📋 Using config-specified target modules: {target_modules}")
        
        lora_config = LoraConfig(
            r=model_config['lora_r'],
            lora_alpha=model_config['lora_alpha'],
            target_modules=target_modules,
            lora_dropout=model_config['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.enable_input_require_grads() 
        
        logger.info(f"🎯 Training mode: LoRA fine-tuning")
        logger.info(f"🔧 LoRA config: r={model_config['lora_r']}, alpha={model_config['lora_alpha']}, dropout={model_config['lora_dropout']}")
        logger.info(f"🎯 Target modules ({len(target_modules)}): {target_modules}")
        
        # 解冻特定层
        # freeze_projector=True 时保护 A 阶段学到的视觉对齐，只训 LoRA + embeddings
        freeze_projector = model_config['freeze_projector']
        if freeze_projector:
            trainable_keywords = ["embed_tokens", "lm_head"]
            logger.info("🔒 Projector FROZEN (freeze_projector=True)")
        else:
            trainable_keywords = ["embed_tokens", "lm_head", "mm_projector", "organ_projectors"]
            logger.info("🔓 Projector TRAINABLE")
        for name, param in model.named_parameters():
            if any(k in name for k in trainable_keywords):
                param.requires_grad = True

    # ================== 核心简化部分 ==================
    # 统一转换模型精度。B200 下直接用 bfloat16。
    # 注意：只转 dtype，不指定 device。device_map="auto" 已将模型分片到多卡，
    # 若指定 device 会尝试把所有参数移到单卡导致 OOM。
    model.to(dtype=torch.bfloat16)
    # mm_projector: nn.Sequential (shared) or nn.Identity (independent/grouped). Move to GPU.
    model.get_model().mm_projector.to(device=device)
    if hasattr(model.get_model(), 'organ_projectors'):
        model.get_model().organ_projectors.to(device=device)
    
    if model_config['gradient_checkpointing']:
        model.gradient_checkpointing_enable()

    # 6. 加载预训练checkpoint中的 LoRA + Embeddings 权重（PEFT wrap 之后）
    # Projector 已在 step 3b 加载（PEFT wrap 之前），这里只处理 LoRA 和 embeddings
    if _ckpt_state_dict is not None:
        state_dict = _ckpt_state_dict
        
        # 提取非 projector 的权重（LoRA + embeddings）
        projector_keywords_filter = ['mm_projector', 'organ_projectors']
        non_projector_keys = {k: v for k, v in state_dict.items() 
                             if not any(kw in k for kw in projector_keywords_filter)}
        
        if non_projector_keys:
            load_result = model.load_state_dict(non_projector_keys, strict=False)
            logger.info(f"📦 Post-PEFT: Loaded non-projector keys ({len(non_projector_keys)} keys)")
        else:
            logger.info(f"📦 Post-PEFT: No LoRA/embedding keys in checkpoint (projector-only checkpoint)")

        # Log load results
        if non_projector_keys and load_result.unexpected_keys:
            logger.info(f"⚠️ {len(load_result.unexpected_keys)} unexpected keys: {load_result.unexpected_keys[:3]}")
        if non_projector_keys and load_result.missing_keys:
            # missing_keys for newly initialized LoRA weights is expected when loading Phase 1 checkpoint
            lora_missing = [k for k in load_result.missing_keys if 'lora_' in k.lower()]
            non_lora_missing = [k for k in load_result.missing_keys if 'lora_' not in k.lower()]
            if non_lora_missing:
                logger.info(f"⚠️ {len(non_lora_missing)} non-LoRA missing keys: {non_lora_missing[:3]}")
    
    # 7. 训练参数统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"📈 Training summary:")
    logger.info(f"   🎯 Total parameters: {total_params:.2f}M")
    logger.info(f"   🔥 Trainable parameters: {trainable_params:.2f}M ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"   📱 Device: {device}, dtype: torch.bfloat16")
    
    return model, tokenizer


def load_model_peft(
    checkpoint_dir: str,
    device: torch.device,
    merge_lora: bool = False,
) -> Tuple[nn.Module, any]:
    """
    Load model from new PEFT folder format checkpoint.

    New format (folder produced by convert_base_checkpoint.py / convert_rag_checkpoint.py):
        step_XXXX/
        ├── adapter_config.json        # LoRA config + _stage3_turn / _rag metadata
        ├── adapter_model.safetensors  # LoRA delta + embed_tokens + lm_head
        ├── tokenizer.json             # Full tokenizer (all special tokens pre-saved)
        ├── projector.pt               # organ_projectors weights (relative keys)
        └── training_state.pt          # step, metrics, optimizer/scheduler (optional)

    Load order (correct — resize before load):
        1. Load tokenizer from checkpoint_dir  → vocab size already correct
        2. Load base LLaVA model
        3. resize_token_embeddings(len(tokenizer))
        4. Build projector structure (from adapter_config._stage3_turn)
        5. PeftModel.from_pretrained → loads LoRA + embed_tokens + lm_head
        6. Load projector.pt → organ_projectors weights
        7. Optionally merge_and_unload LoRA

    Args:
        checkpoint_dir: Path to PEFT folder (e.g. checkpoints_peft/step_6750)
        device: Target device
        merge_lora: If True, merge LoRA weights into base model (faster inference)

    Returns:
        (model, tokenizer)
    """
    from peft import PeftModel
    from transformers import AutoTokenizer
    from llava.model.builder import load_pretrained_model

    logger = get_module_logger()
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"PEFT checkpoint dir not found: {ckpt_dir}")
    # ── 1. Read adapter_config for metadata ──────────────────────────────
    import json
    adapter_cfg_path = ckpt_dir / "adapter_config.json"
    if not adapter_cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {ckpt_dir}")
    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)

    base_model_path = adapter_cfg["base_model_name_or_path"]
    stage3_meta = adapter_cfg["_stage3_turn"]

    projector_type = stage3_meta["projector_type"]
    projector_layers = stage3_meta["projector_layers"]
    tokens_per_embedding = stage3_meta["tokens_per_embedding"]
    whole_ct_dim = stage3_meta["whole_ct_dim"]

    logger.info(f"📂 Loading PEFT checkpoint: {ckpt_dir}")
    logger.info(f"🤖 Base model: {base_model_path}")
    logger.info(f"🔧 Projector: type={projector_type}, layers={projector_layers}, tokens_per_embedding={tokens_per_embedding}")
    

    # ── 2. Load tokenizer from checkpoint_dir (vocab size already correct) ──
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=False)
    logger.info(f"🔤 Tokenizer loaded: vocab_size={len(tokenizer)}")

    # ── 3. Load base LLaVA model (no special tokens yet) ──────────────────
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

    # ── 4. Resize embeddings to match saved tokenizer vocab ───────────────
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    logger.info(f"📐 Resized embeddings to vocab_size={len(tokenizer)}")

    # ── 5. Build projector structure ──────────────────────────────────────
    hidden_size = model.config.hidden_size
    mm_hidden_size = 256

    if whole_ct_dim is not None and whole_ct_dim != mm_hidden_size:
        mm_hidden_sizes = {name: (whole_ct_dim if name == "whole_ct" else mm_hidden_size)
                          for name in ORGAN_TOKEN_NAMES}
    else:
        mm_hidden_sizes = None

    projector_module, _ = build_projectors(
        projector_type=projector_type,
        mm_hidden_size=mm_hidden_size,
        hidden_size=hidden_size,
        projector_layers=projector_layers,
        tokens_per_embedding=tokens_per_embedding,
        logger=logger,
        mm_hidden_sizes=mm_hidden_sizes,
    )

    if projector_type == "shared":
        model.get_model().mm_projector = projector_module
    else:
        model.get_model().mm_projector = nn.Identity()
        model.get_model().organ_projectors = projector_module

    model.config.projector_type = projector_type
    model.config.projector_layers = projector_layers
    model.config.tokens_per_embedding = tokens_per_embedding
    model.config.mm_hidden_sizes = mm_hidden_sizes if mm_hidden_sizes else {name: mm_hidden_size for name in ORGAN_TOKEN_NAMES}

    # ── 6. Load all adapter weights (LoRA + embed_tokens + lm_head) ─────────
    # Strategy: wrap model with PeftModel using adapter_config, then load
    # all weights from safetensors via load_state_dict(strict=False).
    # This avoids key-format issues since our safetensors uses the original
    # PEFT-wrapped key names (base_model.model.model.*) which match exactly
    # what PeftModel.state_dict() expects after wrapping.
    from safetensors.torch import load_file as load_safetensors
    from peft import LoraConfig

    # Build LoraConfig from adapter_config (without modules_to_save — we load those manually)
    lora_config = LoraConfig(
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        target_modules=adapter_cfg["target_modules"],
        lora_dropout=adapter_cfg["lora_dropout"],
        bias=adapter_cfg["bias"],
        task_type="CAUSAL_LM",
    )
    from peft import get_peft_model
    model = get_peft_model(model, lora_config)
    logger.info(f"✅ PeftModel structure built")

    # Load all weights from safetensors into the PEFT-wrapped model
    safetensors_path = ckpt_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        all_weights = load_safetensors(str(safetensors_path), device="cpu")
        load_result = model.load_state_dict(all_weights, strict=False)
        lora_loaded = len([k for k in all_weights if "lora_" in k])
        embed_loaded = len([k for k in all_weights if "embed_tokens" in k or "lm_head" in k])
        logger.info(f"✅ Loaded from safetensors: {lora_loaded} LoRA keys, {embed_loaded} embed/lm_head keys")
        if load_result.unexpected_keys:
            logger.info(f"  ⚠ Unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys[:3]}")
        non_lora_missing = [k for k in load_result.missing_keys if "lora_" not in k]
        if non_lora_missing:
            logger.info(f"  ⚠ Non-LoRA missing keys: {non_lora_missing[:5]}")
    else:
        logger.info(f"⚠️ adapter_model.safetensors not found in {ckpt_dir} — using base weights (projector-only checkpoint)")

    if merge_lora:
        model = model.merge_and_unload()
        logger.info(f"🔀 LoRA merged and unloaded")

    # ── 7. Load projector weights ─────────────────────────────────────────
    projector_pt = ckpt_dir / "projector.pt"
    if projector_pt.exists():
        proj_state = torch.load(projector_pt, map_location="cpu", weights_only=False)
        # proj_state has relative keys like 'lung.0.weight' — rebuild full keys
        if projector_type == "shared":
            load_result = model.get_model().mm_projector.load_state_dict(proj_state, strict=True)
        else:
            load_result = model.get_model().organ_projectors.load_state_dict(proj_state, strict=True)
        logger.info(f"✅ Projector weights loaded ({len(proj_state)} keys)")
    else:
        logger.info(f"⚠️ projector.pt not found in {ckpt_dir} — projector weights are random!")

    # ── 8. Move projector to device + set dtype ───────────────────────────
    model.to(dtype=torch.bfloat16)
    model.get_model().mm_projector.to(device=device)
    if hasattr(model.get_model(), "organ_projectors"):
        model.get_model().organ_projectors.to(device=device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"📈 Total parameters: {total_params:.2f}M")
    logger.info(f"📱 Device: {device}, dtype: torch.bfloat16")

    return model, tokenizer


def build_model_from_peft(
    peft_dir: str,
    config: Dict,
    device: torch.device,
    extra_special_tokens: List[str] = None,
) -> Tuple[nn.Module, any]:
    """
    Build a trainable model by loading a PEFT checkpoint, merging LoRA into base weights,
    then creating a fresh LoRA for downstream training.

    This is the correct way to continue training from a PEFT checkpoint (e.g. E41 → RAG).
    Unlike build_model(), this loads ALL saved weights (LoRA + embed_tokens + lm_head + projector),
    not just the projector.

    Flow:
        1. Read adapter_config.json → base_model_path, projector meta, LoRA config
        2. Load tokenizer from PEFT dir (already has all special tokens from previous training)
        3. Optionally add extra_special_tokens (e.g. RAG tokens for P2_rag)
        4. Load base LLaVA model
        5. resize_token_embeddings to match tokenizer
        6. Build projector structure
        7. Wrap with PeftModel → load adapter_model.safetensors (LoRA + embed_tokens + lm_head)
        8. Merge LoRA into base weights → model is now "pretrained base + fine-tuned"
        9. Load projector.pt
       10. Create fresh LoRA for downstream training
       11. Set trainable params, dtype, device

    Args:
        peft_dir: Path to PEFT checkpoint folder (e.g. E41 step_3500)
        config: Training config dict (needs config["model"] for new LoRA params)
        device: Target device
        extra_special_tokens: Optional list of new tokens to add (e.g. RAG tokens).
                              These must NOT already exist in the tokenizer.

    Returns:
        (model, tokenizer)
    """
    import json
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file as load_safetensors
    from transformers import AutoTokenizer

    logger = get_module_logger()
    ckpt_dir = Path(peft_dir)
    model_config = config["model"]

    # ── 0. Assert checkpoint structure ────────────────────────────────────
    assert ckpt_dir.is_dir(), f"peft_dir must be a directory: {ckpt_dir}"
    adapter_cfg_path = ckpt_dir / "adapter_config.json"
    safetensors_path = ckpt_dir / "adapter_model.safetensors"
    projector_pt_path = ckpt_dir / "projector.pt"
    assert adapter_cfg_path.exists(), f"adapter_config.json not found: {adapter_cfg_path}"
    assert safetensors_path.exists(), f"adapter_model.safetensors not found: {safetensors_path}"
    assert projector_pt_path.exists(), f"projector.pt not found: {projector_pt_path}"

    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)

    base_model_path = adapter_cfg["base_model_name_or_path"]
    stage3_meta = adapter_cfg["_stage3_turn"]
    projector_type = stage3_meta["projector_type"]
    projector_layers = stage3_meta["projector_layers"]
    tokens_per_embedding = stage3_meta["tokens_per_embedding"]
    whole_ct_dim = stage3_meta["whole_ct_dim"]

    logger.info(f"{'='*60}")
    logger.info(f"[build_model_from_peft] Loading PEFT checkpoint: {ckpt_dir}")
    logger.info(f"  base_model   = {base_model_path}")
    logger.info(f"  projector    = type={projector_type}, layers={projector_layers}, "
                f"tokens_per_embedding={tokens_per_embedding}")
    logger.info(f"{'='*60}")

    # ── 1. Load tokenizer from PEFT dir ───────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=False)
    prev_vocab_size = len(tokenizer)
    logger.info(f"Tokenizer loaded from checkpoint: vocab_size={prev_vocab_size}")

    # ── 2. Add extra special tokens (e.g. RAG tokens) ────────────────────
    if extra_special_tokens:
        for tok in extra_special_tokens:
            assert tok not in tokenizer.get_vocab(), (
                f"Token {tok!r} already exists in tokenizer "
                f"(id={tokenizer.convert_tokens_to_ids(tok)}). "
                f"extra_special_tokens should only contain NEW tokens."
            )
        tokenizer.add_special_tokens({"additional_special_tokens": extra_special_tokens})
        logger.info(f"Added {len(extra_special_tokens)} extra tokens: {extra_special_tokens}")
        assert len(tokenizer) == prev_vocab_size + len(extra_special_tokens), (
            f"Vocab size mismatch after adding tokens: "
            f"expected {prev_vocab_size + len(extra_special_tokens)}, got {len(tokenizer)}"
        )

    final_vocab_size = len(tokenizer)
    logger.info(f"Final vocab_size={final_vocab_size}")

    # ── 3. Load base LLaVA model ──────────────────────────────────────────
    logger.info(f"Loading base model: {base_model_path}")
    _, model, _, _ = load_pretrained_model(
        model_path=base_model_path,
        model_base=None,
        model_name="llava-llama",
        load_8bit=False,
        load_4bit=False,
        device_map="auto" if str(device) == "cuda" else str(device),
        device=str(device),
        use_flash_attn=model_config.get("use_flash_attn", False),
    )
    model.config.use_cache = False
    logger.info(f"Base model loaded: hidden_size={model.config.hidden_size}, "
                f"base_vocab_size={model.config.vocab_size}")

    # ── 4. Resize embeddings to match SAVED checkpoint vocab ────────────
    # Must match prev_vocab_size (from saved tokenizer) so load_state_dict
    # doesn't get a size mismatch. Extra tokens are added in step 9 after merge.
    model.resize_token_embeddings(prev_vocab_size, mean_resizing=False)
    assert model.get_input_embeddings().weight.shape[0] == prev_vocab_size
    logger.info(f"Resized embeddings: {model.config.vocab_size} -> {prev_vocab_size}")

    # ── 5. Build projector structure ──────────────────────────────────────
    hidden_size = model.config.hidden_size
    mm_hidden_size = 256

    if whole_ct_dim is not None and whole_ct_dim != mm_hidden_size:
        mm_hidden_sizes = {name: (whole_ct_dim if name == "whole_ct" else mm_hidden_size)
                          for name in ORGAN_TOKEN_NAMES}
    else:
        mm_hidden_sizes = None

    projector_module, _ = build_projectors(
        projector_type=projector_type,
        mm_hidden_size=mm_hidden_size,
        hidden_size=hidden_size,
        projector_layers=projector_layers,
        tokens_per_embedding=tokens_per_embedding,
        logger=logger,
        mm_hidden_sizes=mm_hidden_sizes,
    )

    if projector_type == "shared":
        model.get_model().mm_projector = projector_module
    else:
        model.get_model().mm_projector = nn.Identity()
        model.get_model().organ_projectors = projector_module

    model.config.projector_type = projector_type
    model.config.projector_layers = projector_layers
    model.config.tokens_per_embedding = tokens_per_embedding
    model.config.mm_hidden_sizes = mm_hidden_sizes if mm_hidden_sizes else {
        name: mm_hidden_size for name in ORGAN_TOKEN_NAMES
    }

    # ── 6. Load saved LoRA + embed_tokens + lm_head ──────────────────────
    old_lora_config = LoraConfig(
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        target_modules=adapter_cfg["target_modules"],
        lora_dropout=adapter_cfg["lora_dropout"],
        bias=adapter_cfg["bias"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, old_lora_config)

    all_weights = load_safetensors(str(safetensors_path), device="cpu")
    load_result = model.load_state_dict(all_weights, strict=False)

    lora_keys = [k for k in all_weights if "lora_" in k]
    embed_keys = [k for k in all_weights if "embed_tokens" in k or "lm_head" in k]
    assert len(load_result.unexpected_keys) == 0, (
        f"Unexpected keys when loading safetensors: {load_result.unexpected_keys[:5]}"
    )
    logger.info(f"Loaded safetensors: {len(lora_keys)} LoRA keys, "
                f"{len(embed_keys)} embed/lm_head keys")

    # ── 7. Merge old LoRA into base weights ───────────────────────────────
    model = model.merge_and_unload()
    logger.info(f"Old LoRA merged into base weights")

    # ── 8. Load projector weights ─────────────────────────────────────────
    proj_state = torch.load(projector_pt_path, map_location="cpu", weights_only=False)
    if projector_type == "shared":
        model.get_model().mm_projector.load_state_dict(proj_state, strict=True)
    else:
        model.get_model().organ_projectors.load_state_dict(proj_state, strict=True)
    logger.info(f"Projector weights loaded ({len(proj_state)} keys)")

    # Assert projector is not random
    inner = model.get_model()
    if hasattr(inner, "organ_projectors"):
        sample_key = next(iter(inner.organ_projectors.keys()))
        sample_w = next(inner.organ_projectors[sample_key].parameters())
        assert sample_w.std() > 1e-6, (
            f"Projector weights appear to be zero/random: std={sample_w.std()}"
        )
        logger.info(f"Projector sanity: {sample_key} std={sample_w.float().std().item():.6f}")

    # ── 9. If extra tokens were added, resize embeddings again ────────────
    # After merge_and_unload, embed_tokens has prev_vocab_size rows from the
    # saved checkpoint. If we added extra tokens, we need final_vocab_size rows.
    if extra_special_tokens and final_vocab_size > prev_vocab_size:
        model.resize_token_embeddings(final_vocab_size, mean_resizing=False)
        assert model.get_input_embeddings().weight.shape[0] == final_vocab_size
        logger.info(f"Resized embeddings after merge: {prev_vocab_size} -> {final_vocab_size}")

    # ── 10. Create fresh LoRA for downstream training ─────────────────────
    target_modules = model_config["lora_target_modules"]
    if target_modules is None:
        target_modules = find_all_linear_names(model)
        logger.info(f"Auto-detected target modules: {target_modules}")

    new_lora_config = LoraConfig(
        r=model_config["lora_r"],
        lora_alpha=model_config["lora_alpha"],
        target_modules=target_modules,
        lora_dropout=model_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, new_lora_config)
    model.enable_input_require_grads()
    logger.info(f"Fresh LoRA: r={model_config['lora_r']}, "
                f"alpha={model_config['lora_alpha']}, "
                f"targets={target_modules}")

    # ── 11. Set trainable params ──────────────────────────────────────────
    freeze_projector = model_config["freeze_projector"]
    if freeze_projector:
        trainable_keywords = ["embed_tokens", "lm_head"]
        logger.info("Projector FROZEN")
    else:
        trainable_keywords = ["embed_tokens", "lm_head", "mm_projector", "organ_projectors"]
        logger.info("Projector TRAINABLE")
    for name, param in model.named_parameters():
        if any(k in name for k in trainable_keywords):
            param.requires_grad = True

    # ── 12. dtype + device ────────────────────────────────────────────────
    model.to(dtype=torch.bfloat16)
    model.get_model().mm_projector.to(device=device)
    if hasattr(model.get_model(), "organ_projectors"):
        model.get_model().organ_projectors.to(device=device)

    if model_config.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Total: {total_params:.1f}M, trainable: {trainable_params:.1f}M "
                f"({trainable_params/total_params*100:.1f}%)")
    logger.info(f"dtype=bfloat16, device={device}")
    logger.info(f"[build_model_from_peft] Done")
    logger.info(f"{'='*60}")

    return model, tokenizer
