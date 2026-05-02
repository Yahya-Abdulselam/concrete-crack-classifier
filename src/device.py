"""GPU detection and hardware-aware configuration for PyTorch."""

import os
import platform
import random

import numpy as np
import psutil
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def set_seed(seed: int = config.RANDOM_SEED):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Reproducibility] All seeds set to {seed}")


def detect_gpu() -> tuple:
    """Detect GPU via PyTorch CUDA.

    Returns:
        Tuple of (device, gpu_name, vram_mb).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        # Enable TF32 for faster training on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return device, gpu_name, vram_mb
    return torch.device("cpu"), None, 0


def get_device() -> torch.device:
    """Get the best available device."""
    device, _, _ = detect_gpu()
    return device


def detect_system() -> dict:
    """Get system RAM, CPU cores, and platform info."""
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    cpu_cores = os.cpu_count() or 1
    plat = platform.system()
    return {"ram_gb": ram_gb, "cpu_cores": cpu_cores, "platform": plat}


def get_optimal_config() -> dict:
    """Return optimal batch sizes and worker count based on hardware.

    Returns:
        Dict with batch_sizes (per stage), num_workers, gpu_detected, device.
    """
    device, gpu_name, vram_mb = detect_gpu()
    sys_info = detect_system()

    if vram_mb > 0:
        if vram_mb >= 16000:
            batch_sizes = {1: 64, 2: 64, 3: 32}
        elif vram_mb >= 8000:
            batch_sizes = {1: 32, 2: 32, 3: 16}
        elif vram_mb >= 4000:
            batch_sizes = {1: 16, 2: 16, 3: 8}
        else:
            batch_sizes = {1: 8, 2: 8, 3: 4}
        num_workers = min(8, sys_info["cpu_cores"] // 2)
    else:
        ram_gb = sys_info["ram_gb"]
        if ram_gb >= 32:
            batch_sizes = {1: 16, 2: 16, 3: 8}
        elif ram_gb >= 16:
            batch_sizes = {1: 8, 2: 8, 3: 4}
        else:
            batch_sizes = {1: 4, 2: 4, 3: 2}
        num_workers = min(4, sys_info["cpu_cores"] // 2)

    num_workers = max(num_workers, 0)
    # Windows needs num_workers=0 for DataLoader to avoid multiprocessing issues
    if sys_info["platform"] == "Windows":
        num_workers = 0

    return {
        "batch_sizes": batch_sizes,
        "num_workers": num_workers,
        "gpu_detected": vram_mb > 0,
        "gpu_name": gpu_name,
        "vram_mb": vram_mb,
        "device": device,
    }


def print_device_summary() -> dict:
    """Print formatted hardware summary and return optimal config."""
    cfg = get_optimal_config()
    sys_info = detect_system()

    print(f"\n{'='*60}")
    print("DEVICE SUMMARY")
    print(f"{'='*60}")
    if cfg["gpu_detected"]:
        print(f"  GPU:         {cfg['gpu_name']}")
        print(f"  VRAM:        {cfg['vram_mb']:.0f} MB")
        print(f"  PyTorch:     {torch.__version__} (CUDA {torch.version.cuda})")
    else:
        print("  GPU:         Not detected (CPU-only mode)")
        print(f"  PyTorch:     {torch.__version__}")
    print(f"  RAM:         {sys_info['ram_gb']:.1f} GB")
    print(f"  CPU cores:   {sys_info['cpu_cores']} logical")
    print(f"  Platform:    {sys_info['platform']}")

    print(f"\nRECOMMENDED TRAINING CONFIG")
    print(f"{'-'*40}")
    print(f"  Stage 1 batch size:  {cfg['batch_sizes'][1]}")
    print(f"  Stage 2 batch size:  {cfg['batch_sizes'][2]}")
    print(f"  Stage 3 batch size:  {cfg['batch_sizes'][3]}")
    print(f"  Workers:              {cfg['num_workers']}")
    print(f"{'='*60}")

    return cfg
