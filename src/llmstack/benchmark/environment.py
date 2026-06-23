"""Capture the host environment a benchmark ran on, for reproducibility.

The report records *what hardware produced the numbers* so a reader can judge
them in context. In keeping with llmstack's privacy stance, this deliberately
captures only non-identifying facts (OS, architecture, core count, RAM, GPU) —
never hostname, username, IP, or any other personal identifier.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass

from llmstack import __version__
from llmstack.core.hardware import HardwareProfile, detect_hardware


@dataclass(frozen=True)
class Environment:
    """Non-identifying snapshot of the machine under test."""

    llmstack_version: str
    python_version: str
    os: str
    machine: str
    cpu_cores: int
    ram_gb: float
    gpu_vendor: str
    gpu_name: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary(self) -> str:
        gpu = self.gpu_name or self.gpu_vendor
        return (
            f"llmstack {self.llmstack_version} · Python {self.python_version} · "
            f"{self.os}/{self.machine} · {self.cpu_cores} cores · "
            f"{self.ram_gb:.0f} GB RAM · GPU: {gpu}"
        )


def capture_environment(hardware: HardwareProfile | None = None) -> Environment:
    """Capture the current environment, tolerating hardware-probe failures.

    Pass ``hardware`` to inject a known profile (tests); otherwise it is detected,
    and any detection failure falls back to a minimal CPU-only snapshot.
    """
    if hardware is None:
        try:
            hardware = detect_hardware()
        except Exception:
            hardware = None

    python_version = ".".join(str(p) for p in sys.version_info[:3])
    if hardware is not None:
        return Environment(
            llmstack_version=__version__,
            python_version=python_version,
            os=hardware.os,
            machine=platform.machine(),
            cpu_cores=hardware.cpu_cores,
            ram_gb=round(hardware.ram_gb, 1),
            gpu_vendor=hardware.gpu_vendor,
            gpu_name=hardware.gpu_name,
        )
    return Environment(
        llmstack_version=__version__,
        python_version=python_version,
        os=platform.system().lower(),
        machine=platform.machine(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=0.0,
        gpu_vendor="none",
        gpu_name=None,
    )
