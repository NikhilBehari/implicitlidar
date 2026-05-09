"""GPU selection utility.

Picks the CUDA device with the most free memory (via `nvidia-smi`), with safe
fallbacks to `cuda:0` and finally CPU. Honors `CUDA_VISIBLE_DEVICES` when set.
"""

from __future__ import annotations

import os
import subprocess

import torch


def _free_memory_per_gpu() -> list[int] | None:
    """Return free memory in MiB for every visible GPU, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            stderr=subprocess.DEVNULL,
        )
        return [int(line.strip()) for line in out.decode().strip().splitlines()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def select_device() -> torch.device:
    """Return the best available `torch.device`.

    Logic:
        1. If CUDA is unavailable, return CPU.
        2. If `CUDA_VISIBLE_DEVICES` is set, the runtime has already remapped indices,
           so always return `cuda:0`.
        3. Otherwise, return the CUDA device with the most free memory (or `cuda:0`
           if `nvidia-smi` is not installed).
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return torch.device("cuda:0")
    free = _free_memory_per_gpu()
    if not free:
        return torch.device("cuda:0")
    return torch.device(f"cuda:{free.index(max(free))}")
