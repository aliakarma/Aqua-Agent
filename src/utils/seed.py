"""
src/utils/seed.py
-----------------
Global reproducibility utilities.
Sets seeds for Python, NumPy, PyTorch, and CUDA determinism.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Fix all random seeds for full reproducibility.

    Args:
        seed: Integer seed value (paper uses seeds 42-46 for 5 runs).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # FIX-16 (R1-mn2): Only call CUDA seed functions when a CUDA device is
    # actually available.  Calling them on CPU-only machines is harmless but
    # generates noisy warnings in some PyTorch versions.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic CUDA operations where possible.
    # NOTE: may reduce performance on some GPU kernels.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_rng(seed: int) -> np.random.Generator:
    """Return a private numpy Generator for stochastic components."""
    return np.random.default_rng(seed)
