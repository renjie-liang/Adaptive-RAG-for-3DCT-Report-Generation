"""
Constants for Stage 3 Turn - Organ Tokens and Special Token Indices

This module defines organ-specific tokens that replace generic <image> tokens,
providing semantic meaning for each visual embedding slot.
"""

# Organ token definitions (used in prompts as visual placeholders)
ORGAN_TOKENS = ["<whole_ct>", "<lung>", "<heart>", "<esophagus>", "<aorta>"]
ORGAN_TOKEN_NAMES = ["whole_ct", "lung", "heart", "esophagus", "aorta"]
NUM_ORGAN_TOKENS = len(ORGAN_TOKENS)

# Special token indices for organ tokens (similar to IMAGE_TOKEN_INDEX = -200)
# These negative indices are placeholders that get replaced with actual embeddings
WHOLE_CT_TOKEN_INDEX = -201
LUNG_TOKEN_INDEX = -202
HEART_TOKEN_INDEX = -203
ESOPHAGUS_TOKEN_INDEX = -204
AORTA_TOKEN_INDEX = -205

# Mapping from token string to index
ORGAN_TOKEN_TO_INDEX = {
    "<whole_ct>": WHOLE_CT_TOKEN_INDEX,
    "<lung>": LUNG_TOKEN_INDEX,
    "<heart>": HEART_TOKEN_INDEX,
    "<esophagus>": ESOPHAGUS_TOKEN_INDEX,
    "<aorta>": AORTA_TOKEN_INDEX,
}

# Mapping from index to token string
INDEX_TO_ORGAN_TOKEN = {v: k for k, v in ORGAN_TOKEN_TO_INDEX.items()}

# Mapping from token index to embedding slot index (0-4)
ORGAN_INDEX_TO_SLOT = {
    WHOLE_CT_TOKEN_INDEX: 0,
    LUNG_TOKEN_INDEX: 1,
    HEART_TOKEN_INDEX: 2,
    ESOPHAGUS_TOKEN_INDEX: 3,
    AORTA_TOKEN_INDEX: 4,
}

# All organ token indices as a set for quick lookup
ORGAN_TOKEN_INDICES = set(ORGAN_TOKEN_TO_INDEX.values())

# Provided token for paper samples (no image)
PROVIDED_TOKEN = "<provided>"

# Generation parameters for inference
DEFAULT_GENERATION_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "repetition_penalty": 1.1,
    "max_new_tokens": 512,
}

# Task-specific generation configs
TASK_GENERATION_CONFIG = {
    "report_generation": {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
        "max_new_tokens": 500,
    },
    "long_answer": {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_new_tokens": 300,
    },
    "short_answer": {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 30,
        "repetition_penalty": 1.0,
        "max_new_tokens": 100,
    },
    "multiple_choice": {
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 20,
        "repetition_penalty": 1.0,
        "max_new_tokens": 50,
    },
}
