"""
MedSpatial AI — Hardware Manager
Detects system capabilities and configures optimal inference settings.
Implements memory pressure monitoring and adaptive resolution.
"""

import gc
import os
import platform
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.cuda as cuda


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    has_cuda: bool
    has_mps: bool  # Apple Silicon
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    device: str
    dtype: torch.dtype
    volume_size: int          # Recommended 3D volume size
    half_resolution: bool     # Whether to use 64³ instead of 128³
    amp_enabled: bool         # Automatic mixed precision
    quantize_models: bool     # INT8 quantization for CPU


class HardwareManager:
    """
    Detects hardware and configures the inference pipeline.
    Monitors memory pressure and adapts resolution accordingly.
    """

    MAX_MODEL_MEMORY_GB: float = 2.0
    LOW_MEMORY_THRESHOLD_GB: float = 1.0
    CRITICAL_MEMORY_THRESHOLD_GB: float = 0.5

    def __init__(self) -> None:
        self.profile = self._detect_hardware()
        self._configure_torch()

    def _detect_hardware(self) -> HardwareProfile:
        """Probe the system and return a hardware profile."""
        try:
            import psutil
            ram_total = psutil.virtual_memory().total / 1e9
            ram_available = psutil.virtual_memory().available / 1e9
            cpu_cores = psutil.cpu_count(logical=False) or 2
        except ImportError:
            ram_total = 8.0
            ram_available = 4.0
            cpu_cores = 4

        has_cuda = cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        gpu_name = None
        gpu_vram_gb = None

        if has_cuda:
            try:
                gpu_name = cuda.get_device_name(0)
                gpu_vram_gb = cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                has_cuda = False

        # Determine best device
        if has_cuda:
            device = "cuda"
            dtype = torch.float16 if cuda.get_device_properties(0).major >= 7 else torch.float32
            amp_enabled = True
            quantize_models = False
            # If VRAM < 4GB, use half resolution
            half_resolution = (gpu_vram_gb or 0) < 4.0
        elif has_mps:
            device = "mps"
            dtype = torch.float32
            amp_enabled = False
            quantize_models = False
            half_resolution = False
        else:
            device = "cpu"
            dtype = torch.float32
            amp_enabled = False
            quantize_models = True  # INT8 for CPU
            half_resolution = ram_total < 8.0

        volume_size = 64 if half_resolution else 128

        return HardwareProfile(
            cpu_cores=cpu_cores,
            ram_total_gb=ram_total,
            ram_available_gb=ram_available,
            has_cuda=has_cuda,
            has_mps=has_mps,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            device=device,
            dtype=dtype,
            volume_size=volume_size,
            half_resolution=half_resolution,
            amp_enabled=amp_enabled,
            quantize_models=quantize_models,
        )

    def _configure_torch(self) -> None:
        """Apply global PyTorch settings for current hardware."""
        # Limit CPU thread count to avoid thrashing on low-end machines
        torch.set_num_threads(min(self.profile.cpu_cores, 4))

        # Cap memory on CPU
        if self.profile.device == "cpu":
            os.environ.setdefault("OMP_NUM_THREADS", str(min(self.profile.cpu_cores, 4)))

        # Disable gradient tracking globally during inference
        torch.set_grad_enabled(False)

    def get_device(self) -> torch.device:
        """Return the recommended torch device."""
        return torch.device(self.profile.device)

    def get_dtype(self) -> torch.dtype:
        """Return the recommended dtype for model inference."""
        return self.profile.dtype

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to optimal device with correct dtype."""
        return tensor.to(self.profile.device, dtype=self.profile.dtype, non_blocking=True)

    def check_memory_pressure(self) -> str:
        """
        Check current memory availability.
        Returns: 'ok', 'low', or 'critical'
        """
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / 1e9
            if available_gb < self.CRITICAL_MEMORY_THRESHOLD_GB:
                return "critical"
            elif available_gb < self.LOW_MEMORY_THRESHOLD_GB:
                return "low"
        except ImportError:
            pass
        return "ok"

    def free_memory(self) -> None:
        """Release cached memory."""
        gc.collect()
        if self.profile.has_cuda:
            cuda.empty_cache()
            cuda.synchronize()

    def adaptive_volume_size(self, requested_size: int = 128) -> int:
        """
        Return a potentially reduced volume size based on current memory pressure.
        Falls back to 64³ if memory is low.
        """
        pressure = self.check_memory_pressure()
        if pressure == "critical":
            return min(requested_size, 32)
        elif pressure == "low":
            return min(requested_size, 64)
        return min(requested_size, self.profile.volume_size)

    def get_status_dict(self) -> dict:
        """Return a dict suitable for the /health endpoint."""
        p = self.profile
        try:
            import psutil
            ram_avail = psutil.virtual_memory().available / 1e9
        except ImportError:
            ram_avail = p.ram_available_gb

        status = {
            "device": p.device,
            "cpu_cores": p.cpu_cores,
            "ram_total_gb": round(p.ram_total_gb, 1),
            "ram_available_gb": round(ram_avail, 1),
            "amp_enabled": p.amp_enabled,
            "quantize_models": p.quantize_models,
            "recommended_volume_size": p.volume_size,
            "memory_pressure": self.check_memory_pressure(),
        }
        if p.has_cuda:
            status.update({
                "gpu_name": p.gpu_name,
                "gpu_vram_gb": round(p.gpu_vram_gb or 0, 1),
            })
        return status

    def print_summary(self) -> None:
        """Print detected hardware summary."""
        p = self.profile
        print("=" * 50)
        print("MedSpatial AI — Hardware Profile")
        print("=" * 50)
        print(f"  Platform:  {platform.system()} {platform.machine()}")
        print(f"  Device:    {p.device.upper()}")
        if p.gpu_name:
            print(f"  GPU:       {p.gpu_name} ({p.gpu_vram_gb:.1f} GB VRAM)")
        print(f"  RAM:       {p.ram_total_gb:.1f} GB total, {p.ram_available_gb:.1f} GB free")
        print(f"  Cores:     {p.cpu_cores}")
        print(f"  Volume:    {p.volume_size}³ {'(half-res)' if p.half_resolution else ''}")
        print(f"  AMP:       {'Enabled' if p.amp_enabled else 'Disabled'}")
        print(f"  Quantize:  {'INT8' if p.quantize_models else 'FP32/FP16'}")
        print("=" * 50)


# Global singleton
_hardware_manager: Optional[HardwareManager] = None


def get_hardware_manager() -> HardwareManager:
    """Get or create the global hardware manager instance."""
    global _hardware_manager
    if _hardware_manager is None:
        _hardware_manager = HardwareManager()
    return _hardware_manager
