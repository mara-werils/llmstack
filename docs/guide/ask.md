# Ask Your Files Anything

`llmstack ask` lets you ask questions about local files and directories using a local LLM. No Docker, no server, no config file, no API keys. Just one command.

```bash
llmstack ask "How does authentication work?" ./src/
```

It parses your files, chunks the content, embeds it locally, finds the most relevant sections, and generates an answer with source citations -- all streaming to your terminal in real time.

---

## Installation

```bash
pip install llmstack-cli
```

You also need [Ollama](https://ollama.com) running locally. Install it from the website, then:

```bash
ollama serve
```

That's it. No Docker, no Redis, no Qdrant, no config file. The first time you run `llmstack ask`, it will pull the default models automatically if they are not already available.

---

## Basic Usage

```bash
# Ask about a single file
llmstack ask "Summarize the key findings" report.pdf

# Ask about a directory (recursively scans all supported files)
llmstack ask "How does the auth system work?" ./src/

# Ask about a log file
llmstack ask "What errors occurred after 3pm?" server.log

# Pipe content via stdin
cat contract.pdf | llmstack ask "Are there any risks?"
```

The general syntax:

```
llmstack ask <question> [path] [OPTIONS]
```

- `<question>` -- your question (required)
- `[path]` -- a file or directory to analyze (optional if piping via stdin)

---

## Supported File Types

`llmstack ask` handles a wide range of file types out of the box.

### Documents

| Extension | Type |
|---|---|
| `.pdf` | PDF documents |
| `.docx` | Microsoft Word documents |
| `.md` | Markdown |
| `.txt` | Plain text |
| `.html` | HTML pages |
| `.csv` | CSV data files |

### Code

| Extension | Language |
|---|---|
| `.py` | Python |
| `.js` | JavaScript |
| `.ts` | TypeScript |
| `.go` | Go |
| `.rs` | Rust |
| `.java` | Java |
| `.c`, `.h` | C |
| `.cpp`, `.hpp` | C++ |
| `.rb` | Ruby |
| `.php` | PHP |
| `.swift` | Swift |
| `.kt` | Kotlin |
| `.sh`, `.bash` | Shell scripts |

### Data and Config

| Extension | Type |
|---|---|
| `.json` | JSON |
| `.yaml`, `.yml` | YAML |
| `.toml` | TOML |
| `.xml` | XML |
| `.env` | Environment files |
| `.ini`, `.cfg` | Configuration files |
| `.log` | Log files |

When scanning a directory, `llmstack ask` automatically skips binary files, `node_modules`, `.git`, `__pycache__`, `venv`, and other common non-text directories.

---

## Examples

### Code Analysis

Ask questions about your codebase, find patterns, understand architecture:

```bash
# Understand how a feature works
llmstack ask "How does the rate limiter work?" ./src/gateway/

# Find potential issues
llmstack ask "Find security vulnerabilities" ./src/

# Get an architecture overview
llmstack ask "What is the high-level architecture of this project?" ./src/

# Understand a specific file
llmstack ask "Explain this code step by step" ./src/router/classifier.py

# Find where something is defined
llmstack ask "Where is the database connection configured?" ./src/
```

### PDF and Document Analysis

Analyze reports, contracts, papers, and other documents:

```bash
# Summarize a report
llmstack ask "Summarize the key findings and recommendations" quarterly-report.pdf

# Extract specific information
llmstack ask "What are the payment terms?" contract.pdf

# Compare sections
llmstack ask "What changed between the executive summary and the conclusion?" report.pdf

# Analyze research papers
llmstack ask "What methodology did they use and what are the limitations?" paper.pdf
```

### Log Analysis

Debug issues by asking questions about your logs:

```bash
# Find errors
llmstack ask "What errors occurred and what caused them?" app.log

# Timeline analysis
llmstack ask "What happened between 14:00 and 15:00?" server.log

# Pattern detection
llmstack ask "Are there any repeated failures or patterns?" error.log

# Root cause analysis
llmstack ask "What went wrong before the OOM kill?" syslog
```

### Stdin Piping

Pipe content directly from other commands:

```bash
# Pipe a file
cat README.md | llmstack ask "What does this project do?"

# Pipe command output
git diff HEAD~5 | llmstack ask "Summarize these changes"

# Pipe from curl
curl -s https://example.com/api/docs | llmstack ask "How do I authenticate?"

# Pipe filtered logs
grep "ERROR" app.log | llmstack ask "Categorize these errors"

# Pipe multiple files
cat src/*.py | llmstack ask "What are the main classes and their responsibilities?"
```

---

## Configuration Options

### `--model`

Override the chat model used for generation. Default: `llama3.2`.

```bash
llmstack ask "Explain this code" ./src/ --model llama3.1:70b
llmstack ask "Summarize" report.pdf --model mistral
llmstack ask "Find bugs" ./src/ --model codellama:34b
```

### `--embed-model`

Override the embedding model. Default: `nomic-embed-text`.

```bash
llmstack ask "How does auth work?" ./src/ --embed-model mxbai-embed-large
```

### `--top-k`

Number of most relevant chunks to include in the context. Default: `5`.

```bash
# Include more context for complex questions
llmstack ask "Explain the full request lifecycle" ./src/ --top-k 10

# Use fewer chunks for focused questions
llmstack ask "What port does the server listen on?" ./src/ --top-k 3
```

### `--chunk-size`

Size of text chunks in characters. Default: `1000`.

```bash
# Larger chunks for documents with long sections
llmstack ask "Summarize each chapter" book.md --chunk-size 2000

# Smaller chunks for code files
llmstack ask "Find the bug" ./src/ --chunk-size 500
```

### `--chunk-overlap`

Overlap between adjacent chunks in characters. Default: `200`.

```bash
llmstack ask "Find the bug" ./src/ --chunk-size 800 --chunk-overlap 100
```

### `--no-stream`

Disable streaming output. The full answer is printed at once after generation completes.

```bash
llmstack ask "Summarize this" report.pdf --no-stream
```

---

## How It Works

When you run `llmstack ask`, the following happens:

```
  File(s) ──> Parse ──> Chunk ──> Embed ──> Search ──> Generate
                                    |           |          |
                               Local Ollama   Top-K    Streaming
                               embeddings    nearest   answer with
                                             chunks    citations
```

1. **Parse** -- The target file or directory is scanned. Files are read and parsed based on their type. PDFs are extracted with `pymupdf`, DOCX with `python-docx`, and code/text files are read directly. Binary files and common non-text directories are skipped.

2. **Chunk** -- Parsed text is split into overlapping chunks (default: 1000 characters with 200 character overlap). This ensures that context is not lost at chunk boundaries.

3. **Embed** -- Each chunk is embedded using a local embedding model via Ollama (default: `nomic-embed-text`). All embeddings stay on your machine.

4. **Search** -- Your question is embedded with the same model, and the top-K most similar chunks are retrieved using cosine similarity. This is done in-memory -- no external vector database needed.

5. **Generate** -- The retrieved chunks are assembled into a prompt with your question and sent to the chat model via Ollama. The answer streams to your terminal in real time, with source file and line references.

Everything runs locally. Your files never leave your machine. No data is sent to any external service.

---

## Tips for Best Results

**Be specific with your questions.** "How does auth work?" will get better results than "Tell me about this code." The more specific the question, the better the retrieval step can find relevant chunks.

**Point to the right directory.** If you are asking about authentication, point to `./src/auth/` instead of the entire repo. Fewer, more relevant files means better answers.

**Use a larger model for complex questions.** The default `llama3.2` (3B) is fast and works well for most questions. For deep code analysis or nuanced document interpretation, use `--model llama3.1:70b` or another larger model.

**Increase `--top-k` for broad questions.** If your question spans multiple files or topics, increase `--top-k` to 10 or 15 to include more context.

**Pipe filtered content for large logs.** If a log file is very large, pipe a filtered subset: `grep "ERROR" huge.log | llmstack ask "What went wrong?"`.

**Use stdin for combining sources.** You can concatenate files with `cat` to ask questions across multiple specific files: `cat README.md CONTRIBUTING.md | llmstack ask "How do I contribute?"`.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.11+ | Required for llmstack |
| Ollama | Must be running locally (`ollama serve`) |

That's it. No Docker, no Redis, no Qdrant, no external APIs, no config file. Just Python and Ollama.

---

## Comparison with Full Stack Mode

| | `llmstack ask` | `llmstack up` (Full Stack) |
|---|---|---|
| Use case | Quick file Q&A from terminal | Production API server with full infrastructure |
| Setup | `pip install llmstack-cli` + Ollama | `pip install llmstack-cli` + Docker |
| Config needed | None | `llmstack.yaml` |
| Dependencies | Just Ollama | Docker (Ollama, Qdrant, Redis, etc.) |
| Vector storage | In-memory (per session) | Qdrant (persistent) |
| API server | No | Yes (OpenAI-compatible) |
| Smart routing | No (single model) | Yes (multi-model) |
| Caching | No | Yes (Redis) |
| Web UI | No | Yes |

Use `llmstack ask` when you want a quick answer about local files. Use `llmstack up` when you need a persistent, production-grade API server with RAG, routing, caching, and observability.
