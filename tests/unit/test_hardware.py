"""Tests for hardware detection."""

import subprocess
from unittest.mock import MagicMock, patch

from llmstack.core.hardware import (
    HardwareProfile,
    _check_nvidia_docker,
    _detect_apple,
    _detect_nvidia,
    detect_hardware,
)


def test_detect_hardware_returns_profile():
    hw = detect_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.cpu_cores > 0
    assert hw.ram_mb > 0
    assert hw.os in ("linux", "darwin", "windows")
    assert hw.gpu_vendor in ("nvidia", "amd", "apple", "none")


def _profile(**overrides):
    defaults = dict(
        gpu_vendor="none",
        gpu_name=None,
        gpu_vram_mb=0,
        cpu_cores=4,
        ram_mb=8192,
        os="linux",
        docker_runtime="default",
    )
    defaults.update(overrides)
    return HardwareProfile(**defaults)


def test_profile_has_gpu_false_when_none():
    assert _profile(gpu_vendor="none").has_gpu is False


def test_profile_has_gpu_true_when_present():
    assert _profile(gpu_vendor="nvidia").has_gpu is True


def test_profile_is_apple_silicon():
    assert _profile(gpu_vendor="apple").is_apple_silicon is True
    assert _profile(gpu_vendor="nvidia").is_apple_silicon is False


def test_profile_vram_and_ram_in_gb():
    p = _profile(gpu_vram_mb=2048, ram_mb=16384)
    assert p.gpu_vram_gb == 2.0
    assert p.ram_gb == 16.0


def test_profile_has_nvidia_runtime():
    assert _profile(docker_runtime="nvidia").has_nvidia_runtime is True
    assert _profile(docker_runtime="default").has_nvidia_runtime is False


def test_detect_nvidia_no_binary():
    with patch("llmstack.core.hardware.shutil.which", return_value=None):
        assert _detect_nvidia() == (None, 0)


def test_detect_nvidia_success():
    with (
        patch("llmstack.core.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "llmstack.core.hardware.subprocess.check_output",
            return_value="NVIDIA RTX 4090, 24576\n",
        ),
    ):
        name, vram = _detect_nvidia()
    assert name == "NVIDIA RTX 4090"
    assert vram == 24576


def test_detect_nvidia_empty_output():
    with (
        patch("llmstack.core.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("llmstack.core.hardware.subprocess.check_output", return_value="   "),
    ):
        assert _detect_nvidia() == (None, 0)


def test_detect_nvidia_subprocess_error():
    with (
        patch("llmstack.core.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "llmstack.core.hardware.subprocess.check_output",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5),
        ),
    ):
        assert _detect_nvidia() == (None, 0)


def test_detect_nvidia_value_error_on_bad_parse():
    with (
        patch("llmstack.core.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "llmstack.core.hardware.subprocess.check_output",
            return_value="NVIDIA RTX 4090, not-a-number\n",
        ),
    ):
        assert _detect_nvidia() == (None, 0)


def test_detect_apple_not_darwin():
    with patch("llmstack.core.hardware.platform.system", return_value="Linux"):
        assert _detect_apple() == (None, 0)


def test_detect_apple_darwin_non_apple_chip():
    with (
        patch("llmstack.core.hardware.platform.system", return_value="Darwin"),
        patch("llmstack.core.hardware.subprocess.check_output", return_value="Intel(R) Core(TM)"),
    ):
        assert _detect_apple() == (None, 0)


def test_detect_apple_darwin_apple_silicon():
    fake_mem = MagicMock(total=16 * 1024 * 1024 * 1024)
    with (
        patch("llmstack.core.hardware.platform.system", return_value="Darwin"),
        patch("llmstack.core.hardware.subprocess.check_output", return_value="Apple M2 Pro"),
        patch("llmstack.core.hardware.psutil.virtual_memory", return_value=fake_mem),
    ):
        name, ram_mb = _detect_apple()
    assert name == "Apple M2 Pro"
    assert ram_mb == 16384


def test_detect_apple_subprocess_error():
    with (
        patch("llmstack.core.hardware.platform.system", return_value="Darwin"),
        patch(
            "llmstack.core.hardware.subprocess.check_output",
            side_effect=subprocess.SubprocessError(),
        ),
    ):
        assert _detect_apple() == (None, 0)


def test_check_nvidia_docker_no_binary():
    with patch("llmstack.core.hardware.shutil.which", return_value=None):
        assert _check_nvidia_docker() is False


def test_check_nvidia_docker_runtime_present():
    with (
        patch("llmstack.core.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "llmstack.core.hardware.subprocess.check_output",
            return_value="Runtimes: nvidia runc\n",
        ),
    ):
        assert _check_nvidia_docker() is True


def test_check_nvidia_docker_runtime_absent():
    with (
        patch("llmstack.core.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("llmstack.core.hardware.subprocess.check_output", return_value="Runtimes: runc\n"),
    ):
        assert _check_nvidia_docker() is False


def test_check_nvidia_docker_subprocess_error():
    with (
        patch("llmstack.core.hardware.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "llmstack.core.hardware.subprocess.check_output",
            side_effect=FileNotFoundError(),
        ),
    ):
        assert _check_nvidia_docker() is False


def test_detect_hardware_linux_no_gpu():
    with (
        patch("llmstack.core.hardware.platform.system", return_value="Linux"),
        patch("llmstack.core.hardware._detect_nvidia", return_value=(None, 0)),
        patch("llmstack.core.hardware._detect_apple", return_value=(None, 0)),
    ):
        hw = detect_hardware()
    assert hw.os == "linux"
    assert hw.gpu_vendor == "none"
    assert hw.docker_runtime == "default"


def test_detect_hardware_windows_other_os():
    with (
        patch("llmstack.core.hardware.platform.system", return_value="Windows"),
        patch("llmstack.core.hardware._detect_nvidia", return_value=(None, 0)),
        patch("llmstack.core.hardware._detect_apple", return_value=(None, 0)),
    ):
        hw = detect_hardware()
    assert hw.os == "windows"


def test_detect_hardware_nvidia_with_docker_runtime():
    with (
        patch("llmstack.core.hardware.platform.system", return_value="Linux"),
        patch("llmstack.core.hardware._detect_nvidia", return_value=("RTX 4090", 24576)),
        patch("llmstack.core.hardware._check_nvidia_docker", return_value=True),
    ):
        hw = detect_hardware()
    assert hw.gpu_vendor == "nvidia"
    assert hw.gpu_name == "RTX 4090"
    assert hw.docker_runtime == "nvidia"


def test_detect_hardware_nvidia_without_docker_runtime():
    with (
        patch("llmstack.core.hardware.platform.system", return_value="Linux"),
        patch("llmstack.core.hardware._detect_nvidia", return_value=("RTX 4090", 24576)),
        patch("llmstack.core.hardware._check_nvidia_docker", return_value=False),
    ):
        hw = detect_hardware()
    assert hw.docker_runtime == "default"


def test_detect_hardware_apple_silicon():
    with (
        patch("llmstack.core.hardware.platform.system", return_value="Darwin"),
        patch("llmstack.core.hardware._detect_nvidia", return_value=(None, 0)),
        patch("llmstack.core.hardware._detect_apple", return_value=("Apple M2", 16384)),
    ):
        hw = detect_hardware()
    assert hw.gpu_vendor == "apple"
    assert hw.gpu_name == "Apple M2"
    assert hw.os == "darwin"
