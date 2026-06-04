# Troubleshooting

Common issues and their solutions.

## Connection Errors

### "Connection refused" or "ConnectError"

**Cause:** Ollama is not running or not accessible.

**Fix:**
- Start Ollama: `ollama serve`
- Or start the full stack: `llmstack up`
- Check Ollama is running: `curl http://localhost:11434`

### "Address already in use"

**Cause:** Port 8000 (or another required port) is already occupied.

**Fix:**
- Find the process: `lsof -i :8000`
- Kill it: `kill <PID>`
- Or use a different port: `llmstack serve --port 8001`

## Model Errors

### "model not found"

**Cause:** The requested model hasn't been pulled.

**Fix:**
- Pull the model: `ollama pull llama3.2`
- Or use quickstart: `llmstack quickstart --model llama3.2`
- List available models: `ollama list`

### "CUDA out of memory"

**Cause:** The model is too large for your GPU.

**Fix:**
- Use a smaller model: `llmstack chat --model llama3.2:1b`
- Reduce context: `--max-tokens 512`
- Enable quantization in your config

## Docker Errors

### "Docker daemon is not reachable"

**Cause:** Docker is not running.

**Fix:**
- Start Docker Desktop (macOS/Windows)
- Or: `sudo systemctl start docker` (Linux)
- Check: `docker info`

### GPU not available in Docker

**Cause:** NVIDIA container toolkit not installed.

**Fix:**
- Use the GPU override: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`
- Install toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

## Redis Errors

### "Cannot connect to Redis"

**Cause:** Redis is not running.

**Fix:**
- Start via Docker: `docker run -d -p 6379:6379 redis:7`
- Or start the full stack: `llmstack up`
- Redis is optional — the gateway falls back to in-memory caching

## Configuration

### "No llmstack.yaml found"

**Fix:**
- Create with presets: `llmstack init --preset chat`
- Or quickstart: `llmstack quickstart`

### Config validation error

**Fix:**
- Run `llmstack doctor` for diagnostics
- Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('llmstack.yaml'))"`

## Rate Limiting

### "Rate limit exceeded"

**Cause:** Too many requests in the configured window.

**Fix:**
- Increase limit: `export LLMSTACK_RATE_LIMIT="1000/min"`
- Check current limit in response headers: `X-RateLimit-Limit`
- Wait for `Retry-After` seconds

## Getting Help

- Run `llmstack doctor` for a full system check
- Run `llmstack doctor --fix` to auto-fix common issues
- File an issue: https://github.com/mara-werils/llmstack/issues
