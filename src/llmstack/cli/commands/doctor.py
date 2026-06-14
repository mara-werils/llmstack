"""llmstack doctor — comprehensive system health checks.

Checks Python version, hardware, Docker, Ollama, Redis connectivity,
disk space, GPU availability, network ports, config validity, and
required Python dependencies.  Results are rendered with Rich pass/fail
indicators.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys

import httpx
import psutil

from llmstack.cli.console import console, banner, success, failure, warn, info
from llmstack.core.hardware import detect_hardware
from llmstack.core.preflight import is_port_available, port_owner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_port(port: int) -> bool:
    """Return True if port is available (not in use)."""
    return is_port_available(port)


def _check_url(url: str, timeout: int = 3) -> bool:
    """Check if a URL is reachable."""
    try:
        resp = httpx.get(url, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _check_python_version() -> tuple[int, int]:
    """Return (issues, warnings) for Python version check."""
    issues = warnings = 0
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        success(f"Python {version_str}")
    elif v >= (3, 10):
        warn(f"Python {version_str} (3.11+ recommended)")
        warnings += 1
    else:
        failure(f"Python {version_str} (3.11+ required)")
        issues += 1
    return issues, warnings


def _check_redis() -> tuple[int, int]:
    """Return (issues, warnings) for Redis connectivity."""
    issues = warnings = 0
    redis_url = os.getenv("LLMSTACK_REDIS_URL", "redis://localhost:6379")
    try:
        import redis as _redis

        r = _redis.from_url(redis_url, socket_connect_timeout=2)
        r.ping()
        success(f"Redis is reachable at {redis_url}")
    except ImportError:
        info("redis package not installed (optional, needed for gateway caching)")
    except Exception:
        warn(f"Redis is not reachable at {redis_url}")
        console.print("    [muted]Install: https://redis.io/docs/getting-started/[/]")
        console.print("    [muted]Or: docker run -d -p 6379:6379 redis:7[/]")
        warnings += 1
    return issues, warnings


def _check_disk_space() -> tuple[int, int]:
    """Return (issues, warnings) for available disk space."""
    issues = warnings = 0
    usage = psutil.disk_usage(os.path.expanduser("~"))
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)
    pct_free = 100 - usage.percent
    if free_gb >= 20:
        success(
            f"Disk space: {free_gb:.1f} GB free / {total_gb:.0f} GB total ({pct_free:.0f}% free)"
        )
    elif free_gb >= 5:
        warn(f"Low disk space: {free_gb:.1f} GB free ({pct_free:.0f}% free). Models need space.")
        warnings += 1
    else:
        failure(f"Very low disk space: {free_gb:.1f} GB free. Model downloads may fail.")
        issues += 1
    return issues, warnings


def _check_gpu() -> tuple[int, int]:
    """Return (issues, warnings) for GPU availability."""
    issues = warnings = 0
    hw = detect_hardware()

    if hw.gpu_vendor != "none":
        success(f"GPU: {hw.gpu_name} ({hw.gpu_vram_mb} MB VRAM)")
        if hw.gpu_vendor == "nvidia" and hw.docker_runtime != "nvidia":
            warn("nvidia-container-toolkit not found (GPU passthrough may not work)")
            console.print(
                "    [muted]Install: https://docs.nvidia.com/datacenter/cloud-native/"
                "container-toolkit/install-guide.html[/]"
            )
            warnings += 1
    else:
        # Check for Apple Silicon Metal support
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            success("Apple Silicon detected (Metal acceleration available via Ollama)")
        else:
            warn("No GPU detected (CPU inference only, will be slower)")
            warnings += 1
    return issues, warnings, hw


def doctor() -> None:
    """Check system requirements and diagnose common issues."""
    banner("LLMStack Doctor", "Comprehensive system health check")
    console.print()
    issues = 0
    warnings = 0

    # ── Python ────────────────────────────────────────────────────────
    console.print("[accent]Python[/]")
    i, w = _check_python_version()
    issues += i
    warnings += w
    info(f"Executable: {sys.executable}")

    # ── Hardware ──────────────────────────────────────────────────────
    console.print("\n[accent]Hardware[/]")
    hw = detect_hardware()
    info(f"OS: {hw.os} | CPU: {hw.cpu_cores} cores | RAM: {hw.ram_mb // 1024} GB")

    gpu_result = _check_gpu()
    issues += gpu_result[0]
    warnings += gpu_result[1]
    hw = gpu_result[2]

    recommended_ram = 8
    if hw.ram_mb < recommended_ram * 1024:
        warn(f"Low RAM ({hw.ram_mb // 1024} GB). Recommended: {recommended_ram}+ GB for 7B models")
        warnings += 1

    # ── Disk Space ────────────────────────────────────────────────────
    console.print("\n[accent]Disk Space[/]")
    i, w = _check_disk_space()
    issues += i
    warnings += w

    # ── Docker ────────────────────────────────────────────────────────
    console.print("\n[accent]Docker[/]")
    if shutil.which("docker"):
        success("Docker CLI is installed")
    else:
        failure("Docker is not installed")
        console.print("    [muted]Install: https://docs.docker.com/get-docker/[/]")
        issues += 1

    try:
        import docker

        client = docker.from_env()
        client.ping()
        success("Docker daemon is running")
        docker_info = client.info()
        gpu_runtime = "nvidia" if "nvidia" in str(docker_info.get("Runtimes", {})) else "default"
        info(
            f"Docker version: {docker_info.get('ServerVersion', 'unknown')} "
            f"(runtime: {gpu_runtime})"
        )
    except Exception:
        failure("Docker daemon is not reachable")
        console.print("    [muted]Try: sudo systemctl start docker (or open Docker Desktop)[/]")
        issues += 1

    # ── Ollama ────────────────────────────────────────────────────────
    console.print("\n[accent]Ollama[/]")
    ollama_url = "http://localhost:11434"
    if _check_url(ollama_url):
        success(f"Ollama is running at {ollama_url}")
        try:
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
            models = resp.json().get("models", [])
            if models:
                model_names = [m["name"] for m in models[:5]]
                info(f"Models: {', '.join(model_names)}" + (" ..." if len(models) > 5 else ""))
            else:
                warn("No models pulled. Run: ollama pull llama3.2")
                warnings += 1
        except Exception:
            pass
    else:
        warn("Ollama is not running")
        console.print("    [muted]Install: https://ollama.com/download[/]")
        console.print("    [muted]Start: ollama serve[/]")
        warnings += 1

    # ── Redis ─────────────────────────────────────────────────────────
    console.print("\n[accent]Redis[/]")
    i, w = _check_redis()
    issues += i
    warnings += w

    # ── Network Ports ─────────────────────────────────────────────────
    console.print("\n[accent]Network Ports[/]")
    for port, service in [
        (11434, "Ollama"),
        (6333, "Qdrant"),
        (6379, "Redis"),
        (8000, "Gateway"),
        (8080, "Dashboard"),
    ]:
        if _check_port(port):
            success(f"Port {port} ({service}) is available")
        else:
            if service == "Ollama" and _check_url(f"http://localhost:{port}"):
                success(f"Port {port} ({service}) is in use by {service}")
            else:
                owner = port_owner(port)
                held_by = owner or "another process"
                warn(f"Port {port} ({service}) is in use by {held_by}")
                warnings += 1

    # ── Configuration ─────────────────────────────────────────────────
    console.print("\n[accent]Configuration[/]")
    try:
        from llmstack.config.loader import load_config

        config = load_config()
        success(f"llmstack.yaml is valid (model: {config.models.chat.name})")
    except FileNotFoundError:
        warn("No llmstack.yaml found")
        console.print("    [muted]Run: llmstack init (or llmstack quickstart)[/]")
        warnings += 1
    except SystemExit as exc:
        failure(f"llmstack.yaml validation error: {exc}")
        issues += 1

    # ── Python Dependencies ───────────────────────────────────────────
    console.print("\n[accent]Dependencies[/]")
    for dep in ["typer", "rich", "httpx", "pydantic", "docker", "numpy", "psutil"]:
        try:
            from importlib.metadata import version

            ver = version(dep)
            success(f"{dep} {ver}")
        except Exception:
            failure(f"{dep} is missing")
            issues += 1

    # ── Recommended Models ────────────────────────────────────────────
    console.print("\n[accent]Recommended Models[/]")
    ram_gb = hw.ram_mb // 1024
    vram_mb = hw.gpu_vram_mb

    if vram_mb >= 48000 or ram_gb >= 64:
        info("70B models: llama3.1:70b, qwen2.5:72b (you have plenty of memory)")
    if vram_mb >= 16000 or ram_gb >= 32:
        info("13B-34B models: codellama:34b, deepseek-coder:33b")
    if vram_mb >= 8000 or ram_gb >= 16:
        info("7B-8B models: llama3.2, mistral, codellama:7b (best balance)")
    if ram_gb >= 8:
        info("3B models: llama3.2:3b, phi3:3.8b (fast, good for simple tasks)")
    info("1B models: llama3.2:1b, qwen2.5:1.5b (fastest, great for smart routing)")

    # ── Summary ───────────────────────────────────────────────────────
    console.print()
    if issues == 0 and warnings == 0:
        console.print("[bold green]All checks passed! LLMStack is ready to use.[/]")
    elif issues == 0:
        console.print(f"[bold yellow]{warnings} warning(s), but no blocking issues.[/]")
    else:
        console.print(f"[bold red]{issues} issue(s) and {warnings} warning(s) found.[/]")
    console.print()


def doctor_fix() -> None:
    """Auto-fix common issues detected by doctor."""
    import subprocess

    banner("LLMStack Doctor --fix", "Auto-remediation for common issues")
    console.print()
    fixed = 0

    # 1. Ollama not running
    console.print("[accent]Ollama[/]")
    ollama_url = "http://localhost:11434"
    if not _check_url(ollama_url):
        if shutil.which("ollama"):
            warn("Ollama is not running but binary is available")
            info("Start Ollama with: ollama serve")
            info("Or run in background: nohup ollama serve &")
        else:
            failure("Ollama is not installed")
            info("Install from: https://ollama.com/download")
    else:
        success("Ollama is already running")

        # No models pulled — pull a small one
        console.print("\n[accent]Models[/]")
        try:
            resp = httpx.get(f"{ollama_url}/api/tags", timeout=5)
            models = resp.json().get("models", [])
            if not models:
                warn("No models pulled. Pulling llama3.2:1b ...")
                result = subprocess.run(
                    ["ollama", "pull", "llama3.2:1b"],
                    capture_output=False,
                    check=False,
                )
                if result.returncode == 0:
                    success("Pulled llama3.2:1b")
                    fixed += 1
                else:
                    failure("Failed to pull llama3.2:1b")
            else:
                model_names = [m["name"] for m in models[:5]]
                success(f"Models available: {', '.join(model_names)}")
        except Exception:
            warn("Could not check models")

    # 2. No llmstack.yaml — create from default preset
    console.print("\n[accent]Configuration[/]")
    try:
        from llmstack.config.loader import load_config

        load_config()
        success("llmstack.yaml exists and is valid")
    except FileNotFoundError:
        warn("No llmstack.yaml found — creating from default preset")
        try:
            from llmstack.cli.commands.init import init as _init

            _init(preset="chat", directory=None)
            success("Created llmstack.yaml with 'chat' preset")
            fixed += 1
        except Exception as e:
            failure(f"Failed to create config: {e}")
    except SystemExit:
        warn("llmstack.yaml has validation errors — run 'llmstack init' to recreate")

    # 3. Redis not running
    console.print("\n[accent]Redis[/]")
    redis_url = os.getenv("LLMSTACK_REDIS_URL", "redis://localhost:6379")
    redis_ok = False
    try:
        import redis as _redis

        r = _redis.from_url(redis_url, socket_connect_timeout=2)
        r.ping()
        redis_ok = True
    except ImportError:
        info("redis package not installed (optional)")
    except Exception:
        pass

    if redis_ok:
        success("Redis is reachable")
    elif shutil.which("docker"):
        warn("Redis is not reachable — Docker is available")
        info("Start Redis with: docker run -d -p 6379:6379 redis:7")
    else:
        info("Redis is not reachable and Docker is not installed (Redis is optional)")

    # Summary
    console.print()
    if fixed:
        console.print(f"[bold green]Fixed {fixed} issue(s).[/]")
    else:
        console.print(
            "[bold]No auto-fixable issues found. Run 'llmstack doctor' for full check.[/]"
        )
    console.print()
