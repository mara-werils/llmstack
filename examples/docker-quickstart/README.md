# Docker Quickstart

Get LLMStack running with Ollama in two minutes using Docker Compose.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose v2

## Start

```bash
cd examples/docker-quickstart
docker compose up -d
```

This will:

1. Start an **Ollama** server.
2. Pull the **llama3.2** model automatically.
3. Start the **LLMStack gateway** on port 8000.

## Try it

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Or open the interactive docs at <http://localhost:8000/docs>.

## GPU Support (NVIDIA)

Uncomment the `deploy` section under the `ollama` service in
`docker-compose.yml` to enable GPU passthrough. You need the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
installed.

## Stop

```bash
docker compose down        # stop services
docker compose down -v     # stop and remove data volumes
```
