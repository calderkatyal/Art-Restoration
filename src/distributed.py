"""
Utility script with common distributed functions
"""
import torch
import os

def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return int(os.environ.get("RANK", "0"))

def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))

def get_world_size() -> int:
    """
    Get the world size, the total amount of GPUs.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))

def get_device() -> torch.device:
    """
    Get current rank device.
    """
    return torch.device("cuda", get_local_rank())


def is_main_process() -> bool:
    """True for global rank 0 (single-process runs are rank 0)."""
    return get_global_rank() == 0


