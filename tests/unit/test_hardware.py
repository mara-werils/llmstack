"""Tests for hardware detection."""

from llmstack.core.hardware import detect_hardware, HardwareProfile


def test_detect_hardware_returns_profile():
    hw = detect_hardware()
    assert isinstance(hw, HardwareProfile)
    assert hw.cpu_cores > 0
    assert hw.ram_mb > 0
    assert hw.os in ("linux", "darwin", "windows")
    assert hw.gpu_vendor in ("nvidia", "amd", "apple", "none")
