# Hardware Detection

llmstack automatically detects your hardware and selects the optimal inference backend. This page explains how the detection works and what decisions it makes.

## Detection Process

When you run `llmstack init` or `llmstack up`, the hardware detection module probes your system in this order:

### 1. NVIDIA GPU Detection

llmstack runs `nvidia-smi` to query GPU name and VRAM:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
```

If an NVIDIA GPU is found, it also checks whether the `nvidia-container-toolkit` is installed by inspecting `docker info` for the `nvidia` runtime. This is required for GPU passthrough to Docker containers.

### 2. Apple Silicon Detection

On macOS, llmstack reads the CPU brand string:

```bash
sysctl -n machdep.cpu.brand_string
```

If the result contains "Apple" (indicating M1, M2, M3, or M4 chips), it reports the chip name and uses total system RAM as the unified memory figure, since Apple Silicon shares memory between CPU and GPU.

### 3. CPU Fallback

If neither NVIDIA nor Apple Silicon is detected, llmstack falls back to CPU-only mode. This is fully functional -- Ollama runs GGUF quantized models efficiently on CPU.

## Hardware Profile

The detection result is a `HardwareProfile` with these fields:

| Field | Type | Description |
|---|---|---|
| `gpu_vendor` | `nvidia` \| `apple` \| `amd` \| `none` | Detected GPU vendor |
| `gpu_name` | string or null | GPU model name (e.g., "NVIDIA RTX 4090") |
| `gpu_vram_mb` | integer | GPU VRAM in megabytes (0 for CPU-only) |
| `cpu_cores` | integer | Physical CPU core count |
| `ram_mb` | integer | Total system RAM in megabytes |
| `os` | `linux` \| `darwin` \| `windows` | Operating system |
| `docker_runtime` | `nvidia` \| `default` | Available Docker runtime |

## Backend Selection Rules

The resolver module uses the hardware profile to pick the inference and embedding backends:

### Inference Backend

| Your Hardware | Backend | Reason |
|---|---|---|
| NVIDIA GPU with 16GB+ VRAM | **vLLM** | Maximum throughput with PagedAttention, continuous batching |
| NVIDIA GPU with < 16GB VRAM | **Ollama** | Lower memory overhead, better for smaller GPUs |
| Apple Silicon (M1--M4) | **Ollama** | Metal acceleration for Apple GPUs |
| CPU only | **Ollama** | Efficient GGUF quantized model support |

### Embedding Backend

| Your Hardware | Backend | Reason |
|---|---|---|
| NVIDIA GPU (any) | **TEI** (GPU image) | Hardware-accelerated embeddings |
| Apple Silicon | **TEI** (CPU image) or **Ollama** | TEI provides optimized CPU inference |
| CPU only | **Ollama** | Reuses the inference container |

### Overriding Auto-Detection

You can force a specific backend in `llmstack.yaml`:

```yaml
models:
  chat:
    backend: ollama    # Force Ollama even on NVIDIA 16GB+
  embeddings:
    backend: tei       # Force TEI even on CPU
```

Set `backend: auto` (the default) to let llmstack decide.

You can also control GPU passthrough at the Docker level:

```yaml
docker:
  gpu: "false"    # Disable GPU passthrough entirely
```

## Diagnostics

Use `llmstack doctor` to see what hardware was detected and whether GPU passthrough is working:

```bash
llmstack doctor
```

Example output on an NVIDIA system:

```
LLMStack Doctor

  PASS Docker is installed
  PASS Docker daemon is running
  PASS GPU detected: NVIDIA RTX 4090
  INFO RAM: 64 GB, CPU: 16 cores
  PASS Port 11434 (Ollama) is available
  PASS Port 6333 (Qdrant) is available
  PASS Port 6379 (Redis) is available
  PASS Port 8000 (Gateway) is available

All checks passed!
```

If the nvidia-container-toolkit is missing, you will see:

```
  WARN nvidia-container-toolkit not found (GPU passthrough may not work)
```

In that case, llmstack will still select vLLM as the backend, but the container may fail to start. Install the toolkit or set `docker.gpu: "false"` and `models.chat.backend: ollama` to use CPU inference.
