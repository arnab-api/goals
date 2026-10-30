"""Utilities for managing experiment runtimes and results."""

import logging
import random

import numpy
import torch


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Globally set random seed."""
    logger.info("setting all seeds to %d", seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


