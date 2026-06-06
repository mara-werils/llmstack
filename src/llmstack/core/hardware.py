"""Detect GPU, CPU, and RAM available on the host machine."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

import psutil


@dataclass(frozen=True)
class HardwareProfile:
    gpu_vendor: Literal["nvidia", "amd", "apple", "none"]
    gpu_name: str | None
    gpu_vram_mb: int
    cpu_cores: int
    ram_mb: int
    os: Literal["linux", "darwin", "windows"]
    docker_runtime: Literal["nvidia", "default"]

    @property
    def has_gpu(self) -> bool:
        """Return True when any GPU (NVIDIA, AMD, or Apple Silicon) is available."""
        return self.gpu_vendor != "none"

    @property
    def is_apple_silicon(self) -> bool:
        """Return True when the host uses Apple Silicon."""
        return self.gpu_vendor == "apple"

    @property
    def gpu_vram_gb(self) -> float:
        """Return GPU VRAM in gigabytes."""
        return self.gpu_vram_mb / 1024.0


def _detect_nvidia() -> tuple[str | None, int]:
    """Return (gpu_name, vram_mb) via nvidia-smi, or (None, 0)."""
    if not shutil.which("nvidia-smi"):
        return None, 0
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        if not out:
            return None, 0
        # Take the first GPU
        line = out.splitlines()[0]
        name, vram = line.split(",", 1)
        return name.strip(), int(float(vram.strip()))
    except (subprocess.SubprocessError, ValueError):
        return None, 0


def _detect_apple() -> tuple[str | None, int]:
    """Return (chip_name, unified_memory_mb) on macOS."""
    if platform.system() != "Darwin":
        return None, 0
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
        ).strip()
        # On Apple Silicon, unified memory = total RAM
        ram_bytes = psutil.virtual_memory().total
        if "Apple" in out:
            return out, int(ram_bytes / 1024 / 1024)
        return None, 0
    except (subprocess.SubprocessError, ValueError):
        return None, 0


def _check_nvidia_docker() -> bool:
    """Check if nvidia-container-toolkit is available."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        subprocess.check_output(
            ["docker", "info", "--format", "{{.Runtimes}}"],
            text=True,
            timeout=10,
        )
        # If nvidia runtime exists, docker info will mention it
        out = subprocess.check_output(["docker", "info"], text=True, timeout=10)
        return "nvidia" in out.lower()
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def detect_hardware() -> HardwareProfile:
    """Detect hardware capabilities of the host machine."""
    os_name: Literal["linux", "darwin", "windows"]
    sys = platform.system()
    if sys == "Linux":
        os_name = "linux"
    elif sys == "Darwin":
        os_name = "darwin"
    else:
        os_name = "windows"

    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    ram_mb = int(psutil.virtual_memory().total / 1024 / 1024)

    # Try NVIDIA first
    gpu_name, gpu_vram = _detect_nvidia()
    if gpu_name:
        has_nvidia_docker = _check_nvidia_docker()
        return HardwareProfile(
            gpu_vendor="nvidia",
            gpu_name=gpu_name,
            gpu_vram_mb=gpu_vram,
            cpu_cores=cpu_cores,
            ram_mb=ram_mb,
            os=os_name,
            docker_runtime="nvidia" if has_nvidia_docker else "default",
        )

    # Try Apple Silicon
    gpu_name, gpu_vram = _detect_apple()
    if gpu_name:
        return HardwareProfile(
            gpu_vendor="apple",
            gpu_name=gpu_name,
            gpu_vram_mb=gpu_vram,
            cpu_cores=cpu_cores,
            ram_mb=ram_mb,
            os=os_name,
            docker_runtime="default",
        )

    # No GPU detected
    return HardwareProfile(
        gpu_vendor="none",
        gpu_name=None,
        gpu_vram_mb=0,
        cpu_cores=cpu_cores,
        ram_mb=ram_mb,
        os=os_name,
        docker_runtime="default",
    )
