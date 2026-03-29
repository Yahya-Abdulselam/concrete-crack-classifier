"""Device detection and auto-configuration for training optimization."""

import os
import platform


def detect_gpu() -> dict | None:
    """Detect GPU via TensorFlow. Returns dict with name/vram_gb or None."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Enable memory growth to avoid pre-allocating all VRAM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            details = tf.config.experimental.get_device_details(gpus[0])
            name = details.get("device_name", str(gpus[0]))

            # Try to get VRAM from device details
            vram_gb = None
            if "compute_capability" in details:
                # TF doesn't directly expose VRAM; estimate from nvidia-smi
                vram_gb = _get_vram_from_smi()

            return {"name": name, "vram_gb": vram_gb, "count": len(gpus)}
    except Exception:
        pass

    # TF couldn't find GPU — print diagnostic
    print("=" * 60)
    print("WARNING: TensorFlow cannot detect GPU.")
    print("  PyTorch may see the GPU, but TF needs matching CUDA/cuDNN.")
    print("  Diagnostics:")
    print("    1. Verify CUDA 12.x is installed (not just the NVIDIA driver)")
    print("    2. Verify cuDNN 9.x is installed and on PATH")
    print("    3. Run: python -c \"import tensorflow as tf; "
          "print(tf.sysconfig.get_build_info())\"")
    print("  Training will proceed on CPU (significantly slower).")
    print("=" * 60)
    return None


def _get_vram_from_smi() -> float | None:
    """Try to get GPU VRAM in GB from nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            mb = float(result.stdout.strip().split("\n")[0])
            return round(mb / 1024, 1)
    except Exception:
        pass
    return None


def detect_system() -> dict:
    """Detect system RAM and CPU cores."""
    info = {
        "cpu_cores_logical": os.cpu_count() or 4,
        "ram_gb": None,
        "platform": platform.system(),
        "is_windows": platform.system() == "Windows",
    }

    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    return info


def get_optimal_config(vram_gb: float | None, ram_gb: float | None, num_cores: int) -> dict:
    """Return optimal per-stage batch sizes and worker count based on hardware.

    Args:
        vram_gb: GPU VRAM in GB, or None if no GPU.
        ram_gb: System RAM in GB, or None if unknown.
        num_cores: Logical CPU core count.

    Returns:
        Dict with 'batch_sizes' (per-stage dict) and 'num_workers'.
    """
    # Per-stage batch sizes based on VRAM
    if vram_gb is None:
        # CPU-only — keep batches small
        batch_sizes = {1: 8, 2: 8, 3: 4}
    elif vram_gb <= 4:
        # 4GB GPU (e.g., RTX 3050 Laptop)
        batch_sizes = {1: 16, 2: 16, 3: 8}
    elif vram_gb <= 6:
        # 6GB GPU (e.g., RTX 2060, GTX 1660)
        batch_sizes = {1: 24, 2: 16, 3: 12}
    elif vram_gb <= 8:
        # 8GB GPU (e.g., RTX 3070, RTX 4060)
        batch_sizes = {1: 32, 2: 32, 3: 16}
    else:
        # 12GB+ GPU (e.g., RTX 3080+, A100)
        batch_sizes = {1: 32, 2: 32, 3: 32}

    # Workers: cap at cores/4, max 8, min 1
    # On Windows use_multiprocessing=False so these are threads
    num_workers = max(1, min(num_cores // 4, 8))
    if vram_gb is None:
        # CPU-only: fewer workers to leave cores for training
        num_workers = max(1, min(num_cores // 8, 4))

    return {
        "batch_sizes": batch_sizes,
        "num_workers": num_workers,
    }


def print_device_summary() -> dict:
    """Detect hardware and print a formatted summary. Returns the optimal config."""
    gpu_info = detect_gpu()
    sys_info = detect_system()

    vram_gb = gpu_info["vram_gb"] if gpu_info else None
    ram_gb = sys_info["ram_gb"]
    num_cores = sys_info["cpu_cores_logical"]

    config = get_optimal_config(vram_gb, ram_gb, num_cores)

    print("\n" + "=" * 60)
    print("DEVICE SUMMARY")
    print("=" * 60)

    if gpu_info:
        print(f"  GPU:         {gpu_info['name']}")
        if vram_gb:
            print(f"  VRAM:        {vram_gb} GB")
        print(f"  GPU count:   {gpu_info['count']}")
    else:
        print("  GPU:         Not detected (CPU-only mode)")

    if ram_gb:
        print(f"  RAM:         {ram_gb} GB")
    print(f"  CPU cores:   {num_cores} logical")
    print(f"  Platform:    {sys_info['platform']}")

    print("\nRECOMMENDED TRAINING CONFIG")
    print("-" * 40)
    for stage, bs in config["batch_sizes"].items():
        print(f"  Stage {stage} batch size:  {bs}")
    print(f"  Workers:              {config['num_workers']}")
    print(f"  Multiprocessing:      False" if sys_info["is_windows"]
          else f"  Multiprocessing:      True")
    print("=" * 60 + "\n")

    return config
