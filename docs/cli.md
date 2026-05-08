# CLI Reference

llmstack provides a command-line interface built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/). All commands are available under the `llmstack` binary after installation.

## `llmstack init`

Initialize a new `llmstack.yaml` configuration file with smart defaults.

```bash
llmstack init [OPTIONS]
```

**Options**

| Option | Short | Default | Description |
|---|---|---|---|
| `--preset` | `-p` | None | Preset to use: `chat`, `rag`, or `agent` |
| `--dir` | `-d` | Current directory | Directory to create the config file in |

**What It Does**

1. Detects your hardware (GPU, CPU, RAM)
2. Selects the optimal inference backend based on your GPU
3. Applies the selected preset (or defaults)
4. Writes `llmstack.yaml` to the target directory

**Examples**

```bash
# Default configuration with auto-detected settings
llmstack init

# Use the RAG preset
llmstack init --preset rag

# Create config in a specific directory
llmstack init --dir ~/my-project

# Use the agent preset for heavy workloads
llmstack init --preset agent
```

**Presets**

| Preset | Description |
|---|---|
| `chat` | Minimal setup: inference + cache + gateway. No vector DB or embeddings. |
| `rag` | Full RAG setup: adds Qdrant and TEI for document ingestion and semantic search. |
| `agent` | Heavy-duty: 70B model, 16K context length, extended timeouts. |

If the config file already exists, the command exits with an error. Use `--dir` to create it in a different location.

---

## `llmstack up`

Start all services defined in `llmstack.yaml`.

```bash
llmstack up [OPTIONS]
```

**Options**

| Option | Short | Default | Description |
|---|---|---|---|
| `--attach` | `-a` | `false` | Stream inference logs after starting |

**What It Does**

1. Loads and validates `llmstack.yaml`
2. Detects hardware and resolves backends
3. Creates the Docker network
4. Builds the gateway Docker image (if needed)
5. Generates an API key (if `auth: api_key` and no keys exist)
6. Starts services in dependency order with health checks:
    - Qdrant (vector DB)
    - Redis (cache)
    - Inference (Ollama or vLLM)
    - Embeddings (TEI or Ollama)
    - Gateway (FastAPI)
    - Prometheus (metrics)
    - Grafana (dashboard)
7. Pulls the configured model (e.g., `ollama pull llama3.2`)
8. Prints a summary table with service URLs

**Examples**

```bash
# Start all services
llmstack up

# Start and follow inference logs
llmstack up --attach
```

---

## `llmstack down`

Stop and remove all llmstack services.

```bash
llmstack down [OPTIONS]
```

**Options**

| Option | Short | Default | Description |
|---|---|---|---|
| `--volumes` | `-v` | `false` | Also remove data volumes (model cache, vector data, etc.) |

**Examples**

```bash
# Stop services, keep data
llmstack down

# Stop services and delete all data
llmstack down --volumes
```

!!! warning
    Using `--volumes` permanently deletes downloaded models, cached responses, and stored vectors. Use with caution.

---

## `llmstack status`

Show the health status of all running llmstack services.

```bash
llmstack status
```

**Output**

A Rich-formatted table showing:

| Column | Description |
|---|---|
| Service | Service name (e.g., `ollama`, `qdrant`) |
| Container | Docker container ID |
| Status | `running` (green) or `stopped` (red) |
| Ports | Host-to-container port mappings |

**Example Output**

```
       LLMStack Status
┌────────────┬──────────────┬─────────┬──────────────────┐
│ Service    │ Container    │ Status  │ Ports            │
├────────────┼──────────────┼─────────┼──────────────────┤
│ qdrant     │ abc123def456 │ running │ 6333->6333/tcp   │
│ redis      │ def456ghi789 │ running │ 6379->6379/tcp   │
│ ollama     │ ghi789jkl012 │ running │ 11434->11434/tcp │
│ gateway    │ jkl012mno345 │ running │ 8000->8000/tcp   │
│ prometheus │ mno345pqr678 │ running │ 9090->9090/tcp   │
│ grafana    │ pqr678stu901 │ running │ 8080->3000/tcp   │
└────────────┴──────────────┴─────────┴──────────────────┘
```

If no services are running, it prints a hint to run `llmstack up`.

---

## `llmstack chat`

Start an interactive terminal chat session with the running LLM.

```bash
llmstack chat [OPTIONS]
```

**Options**

| Option | Default | Description |
|---|---|---|
| `--model` | Config default | Override the model to chat with |

**What It Does**

1. Loads `llmstack.yaml` to find the gateway URL and API key
2. Verifies the gateway is reachable via `/healthz`
3. Opens an interactive prompt
4. Sends messages to `/v1/chat/completions` with streaming enabled
5. Displays streamed tokens in real time

**In-Session Commands**

| Command | Description |
|---|---|
| `/clear` | Clear conversation history |
| `exit` or `quit` | End the session |
| `Ctrl+C` | End the session |

**Example**

```bash
llmstack chat
```

```
LLMStack Chat -- model: llama3.2
Type 'exit' or Ctrl+C to quit. '/clear' to reset conversation.

You: What is the capital of France?
Assistant: The capital of France is Paris.

You: What is it known for?
Assistant: Paris is known for the Eiffel Tower, the Louvre Museum,
Notre-Dame Cathedral, and its rich culture, cuisine, and history.

You: /clear
Conversation cleared.

You: exit
Goodbye!
```

---

## `llmstack export`

Generate a standalone `docker-compose.yml` from the current configuration.

```bash
llmstack export [OPTIONS]
```

**Options**

| Option | Default | Description |
|---|---|---|
| `--output` | `docker-compose.yml` | Output file path |

**What It Does**

1. Loads `llmstack.yaml`
2. Detects hardware and resolves backends
3. Generates a complete `docker-compose.yml` with all services, volumes, networks, and environment variables
4. Writes the file to disk

The exported file is standalone -- recipients do not need llmstack installed. They just run `docker compose up -d`.

**Examples**

```bash
# Export to default file
llmstack export

# Export to a custom path
llmstack export --output my-stack.yml
```

**Example Output**

```
Exported 7 services to docker-compose.yml
Run with: docker compose -f docker-compose.yml up -d
```

---

## `llmstack logs`

Stream logs from a specific service.

```bash
llmstack logs SERVICE [OPTIONS]
```

**Arguments**

| Argument | Description |
|---|---|
| `SERVICE` | Service name: `ollama`, `vllm`, `qdrant`, `redis`, `gateway`, `tei`, `prometheus`, `grafana` |

**Options**

| Option | Short | Default | Description |
|---|---|---|---|
| `--follow/--no-follow` | `-f` | `true` | Follow log output in real time |
| `--tail` | `-n` | `50` | Number of initial lines to show |

**Examples**

```bash
# Follow Ollama logs
llmstack logs ollama

# Show last 100 lines of gateway logs without following
llmstack logs gateway --no-follow --tail 100

# Follow Redis logs
llmstack logs redis
```

Press `Ctrl+C` to stop following.

---

## `llmstack doctor`

Diagnose common issues with your system and configuration.

```bash
llmstack doctor
```

**What It Checks**

| Check | Pass | Fail/Warn |
|---|---|---|
| Docker installed | `docker` binary found | Docker not installed |
| Docker daemon | `docker.ping()` succeeds | Daemon not reachable |
| GPU detection | GPU vendor and name detected | No GPU (CPU inference only) |
| nvidia-container-toolkit | `nvidia` runtime in `docker info` | Toolkit not found (GPU passthrough may fail) |
| System resources | RAM and CPU core count | Informational only |
| Port availability | Ports 11434, 6333, 6379, 8000 are free | Port already in use |
| Config validation | `llmstack.yaml` parses successfully | Validation errors or file not found |

**Example Output**

```
LLMStack Doctor

  PASS Docker is installed
  PASS Docker daemon is running
  WARN No GPU detected (CPU inference only)
  INFO RAM: 16 GB, CPU: 8 cores
  PASS Port 11434 (Ollama) is available
  PASS Port 6333 (Qdrant) is available
  PASS Port 6379 (Redis) is available
  PASS Port 8000 (Gateway) is available
  PASS llmstack.yaml is valid

All checks passed!
```
