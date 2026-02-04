"""
Shared test fixtures and utilities for the goals project.

This module provides reusable initialization code for tests:
- Model loading with a smaller test model
- Dataset loading
- Common test utilities

Usage:
    from test.fixtures import get_test_model, get_test_dataset, get_test_sample
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables before importing torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from src.models import ModelandTokenizer
from src.goals.data import IncompleteCodeDataset

# Default test model - smaller model for faster testing
DEFAULT_TEST_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"

# Cached instances to avoid reloading
_cached_model = None
_cached_dataset = None


def get_test_model(
    model_key: str = DEFAULT_TEST_MODEL,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    force_reload: bool = False,
) -> ModelandTokenizer:
    """
    Load and cache a test model.

    Args:
        model_key: HuggingFace model identifier
        dtype: Model dtype (default: bfloat16)
        device_map: Device mapping strategy (default: "auto")
        force_reload: If True, reload even if cached

    Returns:
        ModelandTokenizer instance
    """
    global _cached_model

    if _cached_model is None or force_reload:
        print(f"Loading test model: {model_key}")
        _cached_model = ModelandTokenizer(
            model_key=model_key,
            dtype=dtype,
            device_map=device_map,
            attn_implementation="eager",
        )
        print(f"Model loaded on device: {_cached_model.device}")

    return _cached_model


def get_test_dataset(force_reload: bool = False) -> IncompleteCodeDataset:
    """
    Load and cache the evaluation dataset.

    Args:
        force_reload: If True, reload even if cached

    Returns:
        IncompleteCodeDataset instance
    """
    global _cached_dataset

    if _cached_dataset is None or force_reload:
        _cached_dataset = IncompleteCodeDataset.load_dataset()
        print(f"Loaded {len(_cached_dataset)} samples")

    return _cached_dataset


def get_test_sample(index: int = 0):
    """
    Get a specific sample from the test dataset.

    Args:
        index: Sample index (default: 0, the is_odd function)

    Returns:
        IncompleteCodeSample instance
    """
    dataset = get_test_dataset()
    return dataset[index]


def create_test_prompt(
    mt: ModelandTokenizer,
    sample,
    prefill: str = "",
) -> str:
    """
    Create a properly formatted prompt for testing.

    Args:
        mt: ModelandTokenizer instance
        sample: IncompleteCodeSample instance
        prefill: Optional prefill string to append

    Returns:
        Formatted prompt string
    """
    from src.goals.functional import supports_chat_template, apply_chat_template

    if supports_chat_template(mt):
        return apply_chat_template(
            mt=mt,
            user_query=sample.instruction.strip(),
            incomplete_code=sample.code + prefill,
        )
    else:
        return sample.instruction + "<code>" + sample.code + prefill


def cleanup():
    """Clear cached instances to free memory."""
    global _cached_model, _cached_dataset
    _cached_model = None
    _cached_dataset = None
    torch.cuda.empty_cache()
