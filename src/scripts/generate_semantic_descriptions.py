"""Original executable entry point; implementation lives in semantic/."""
import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import argparse
import json
import os
import pickle
import re
import shutil
import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from semantic.prompts import (
    PROMPT_VERSION,
    MODALITY_PROMPT_VERSIONS,
    TEXT_RAW_SYSTEM,
    TEXT_RAW_EXAMPLE_USER,
    TEXT_RAW_EXAMPLE_ASSISTANT,
    TONE_LABELS,
    TONE_ALIASES,
    RAW_PROMPTS,
    INVALID_VISION_RESPONSES,
    INVALID_META_RESPONSES,
    DEFAULT_MODEL_PATH,
)
from semantic.checkpoints import (
    read_json_checkpoint,
    atomic_write_json,
)
from semantic.inputs import (
    load_inputs,
    parse_sample_id,
    segment_sort_key,
    build_text_contexts,
)
from semantic.cleaning import (
    clean_model_response,
    validate_and_normalize_description,
    empty_observation_description,
)
from semantic.storage import (
    load_or_initialize_output,
    ensure_raw_storage,
    ensure_fallback_storage,
    ensure_output_ids,
    reset_all_descriptions,
    reset_selected_descriptions,
    sanitize_existing_descriptions,
    has_modality_descriptions,
    has_any_description,
    reset_invalid_vision_descriptions,
    reset_empty_fallback_descriptions,
)
from semantic.generation import (
    generate_single_pass_description,
    generate,
    extract_video_frames,
    generate_from_messages,
)
from semantic.cli import parse_args, main

def __getattr__(name):
    # Diagnostic compatibility for checkpoint counters; ownership is centralized.
    if name in {'_ATOMIC_WRITE_COUNT', '_BACKUP_INTERVAL'}:
        from semantic import checkpoints
        return getattr(checkpoints, name)
    raise AttributeError(name)

if __name__ == '__main__':
    main()
