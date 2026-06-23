"""Tests for benchmark environment capture (llmstack.benchmark.environment)."""

from __future__ import annotations

from llmstack.benchmark import environment as env_mod
from llmstack.benchmark.environment import Environment, capture_environment
from llmstack.core.hardware import HardwareProfile


def _fake_hw(**over) -> HardwareProfile:
    base = dict(
        gpu_vendor="apple",
        gpu_name="Apple M3",
        gpu_vram_mb=16384,
        cpu_cores=8,
        ram_mb=16384,
        os="darwin",
        docker_runtime="default",
    )
    base.update(over)
    return HardwareProfile(**base)


def test_capture_with_injected_hardware() -> None:
    env = capture_environment(hardware=_fake_hw())
    assert env.os == "darwin"
    assert env.cpu_cores == 8
    assert env.ram_gb == 16.0
    assert env.gpu_vendor == "apple"
    assert env.gpu_name == "Apple M3"
    assert env.llmstack_version
    assert env.python_version.count(".") == 2


def test_summary_is_one_line_and_non_identifying() -> None:
    env = capture_environment(hardware=_fake_hw())
    s = env.summary()
    assert "\n" not in s
    assert "Apple M3" in s
    # No identifying fields are ever present in the dataclass.
    keys = set(env.as_dict())
    assert "hostname" not in keys
    assert "user" not in keys
    assert "ip" not in keys


def test_detect_failure_falls_back_to_cpu_only(monkeypatch) -> None:
    def _boom():
        raise RuntimeError("no hardware probe")

    monkeypatch.setattr(env_mod, "detect_hardware", _boom)
    env = capture_environment()
    assert isinstance(env, Environment)
    assert env.gpu_vendor == "none"
    assert env.cpu_cores >= 1
    assert env.gpu_name is None


def test_real_detect_path_does_not_raise(monkeypatch) -> None:
    # Inject a profile via the detect function to avoid spawning subprocesses.
    monkeypatch.setattr(
        env_mod, "detect_hardware", lambda: _fake_hw(gpu_vendor="none", gpu_name=None)
    )
    env = capture_environment()
    assert env.gpu_vendor == "none"


def test_as_dict_roundtrip_keys() -> None:
    env = capture_environment(hardware=_fake_hw())
    d = env.as_dict()
    assert set(d) == {
        "llmstack_version",
        "python_version",
        "os",
        "machine",
        "cpu_cores",
        "ram_gb",
        "gpu_vendor",
        "gpu_name",
    }
