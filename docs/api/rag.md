# RAG API

The RAG (Retrieval-Augmented Generation) endpoints allow you to ingest documents, query them with semantic search, and generate answers grounded in your data.

## Base URL

```
http://localhost:8000/v1/rag
```

## Authentication

All RAG endpoints require a Bearer token (same as the OpenAI-compatible endpoints):

```
Authorization: Bearer sk-llmstack-...
```

## Ingest Documents

Chunk, embed, and store a document in the vector database.

### `POST /v1/rag/ingest`

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `text` | string | Yes | The document text to ingest |
| `source` | string | Yes | A source identifier (e.g., filename, URL). Used for deduplication and deletion. |

**How Ingestion Works**

1. The text is split into chunks of approximately 512 words with a 64-word overlap between consecutive chunks
2. Each chunk is embedded using the configured embedding model
3. Chunk IDs are deterministic (based on a hash of the content), so re-ingesting the same document updates existing vectors rather than creating duplicates
4. Vectors are stored in Qdrant with metadata (source, chunk index, text)

**Example Request**

```bash
curl http://localhost:8000/v1/rag/ingest \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "LLMStack is an open-source tool that deploys a complete LLM infrastructure with a single command. It includes inference, embeddings, vector storage, caching, and monitoring.",
    "source": "docs.txt"
  }'
```

**Example Response**

```json
{
  "status": "ok",
  "chunks": 1,
  "source": "docs.txt"
}
```

**Ingesting Large Documents**

For large documents, the chunking happens server-side. Send the entire document text in a single request. The gateway handles splitting, embedding, and storage.

```python
import httpx

text = open("large-document.txt").read()

response = httpx.post(
    "http://localhost:8000/v1/rag/ingest",
    json={"text": text, "source": "large-document.txt"},
    headers={"Authorization": "Bearer YOUR_KEY"},
    timeout=120,  # Large documents may take longer
)
```

## Query with RAG

Perform semantic search over ingested documents and generate an answer.

### `POST /v1/rag/query`

**Request Body**

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | The question to answer |
| `top_k` | integer | No | Number of chunks to retrieve (default: 5) |
| `stream` | boolean | No | Enable SSE streaming for the generated answer (default: `false`) |

**How Querying Works**

1. The question is embedded using the same embedding model used for ingestion
2. A semantic search is performed in Qdrant to find the `top_k` most relevant chunks
3. The retrieved chunks are assembled into a context prompt
4. The chat model generates an answer grounded in the retrieved context
5. Source citations are included in the response

**Example Request**

```bash
curl http://localhost:8000/v1/rag/query \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is LLMStack?",
    "top_k": 5
  }'
```

**Example Response**

```json
{
  "answer": "LLMStack is an open-source tool that deploys a complete LLM infrastructure with a single command. It includes inference, embeddings, vector storage, caching, and monitoring.",
  "sources": [
    {
      "source": "docs.txt",
      "chunk_index": 0,
      "score": 0.95,
      "text": "LLMStack is an open-source tool that deploys..."
    }
  ]
}
```

### Streaming

When `stream: true`, the answer is streamed via Server-Sent Events while the sources are included in the final event:

```bash
curl http://localhost:8000/v1/rag/query \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is LLMStack?",
    "stream": true
  }'
```

```
data: {"token": "LLMStack"}
data: {"token": " is"}
data: {"token": " an"}
data: {"token": " open-source"}
...
data: {"token": "", "sources": [{"source": "docs.txt", "score": 0.95}]}
data: [DONE]
```

## Delete Documents

Remove all chunks associated with a source.

### `DELETE /v1/rag/documents/{source}`

**Path Parameters**

| Parameter | Type | Description |
|---|---|---|
| `source` | string | The source identifier used during ingestion |

**Example Request**

```bash
curl -X DELETE http://localhost:8000/v1/rag/documents/docs.txt \
  -H "Authorization: Bearer YOUR_KEY"
```

**Example Response**

```json
{
  "status": "ok",
  "deleted": 3,
  "source": "docs.txt"
}
```

## Collection Status

Get statistics about the RAG collection.

### `GET /v1/rag/status`

**Example Request**

```bash
curl http://localhost:8000/v1/rag/status \
  -H "Authorization: Bearer YOUR_KEY"
```

**Example Response**

```json
{
  "collection": "llmstack",
  "points": 1234,
  "vectors_size": 1024,
  "status": "green"
}
```

| Field | Description |
|---|---|
| `collection` | Qdrant collection name |
| `points` | Total number of stored vectors (chunks) |
| `vectors_size` | Embedding dimension size |
| `status` | Collection health (`green`, `yellow`, `red`) |

## Tips

**Chunk size**: The default chunking (512 words, 64-word overlap) works well for most documents. If your documents have very short paragraphs, consider concatenating them before ingestion.

**Deduplication**: Chunk IDs are deterministic. Re-ingesting a document with the same content is safe -- existing vectors are updated, not duplicated.

**top_k tuning**: Start with `top_k: 5` (the default). Increase it if the model needs more context to answer accurately. Decrease it if responses are slow or include irrelevant information.

**Source management**: Use meaningful source identifiers. The `DELETE` endpoint deletes by source, so grouping related content under one source makes cleanup easier.
