# Python SDK

llmstack exposes an OpenAI-compatible API, which means you can use any OpenAI client library to interact with it. This page covers common usage patterns with Python.

## Using the OpenAI Python SDK

The official `openai` package works out of the box with llmstack. Just point it to your gateway URL:

```bash
pip install openai
```

### Chat Completions

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",  # From llmstack.yaml gateway.api_keys
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms."},
    ],
)

print(response.choices[0].message.content)
```

### Streaming Chat Completions

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",
)

stream = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Write a haiku about programming."}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

### Embeddings

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",
)

response = client.embeddings.create(
    model="bge-m3",
    input=["Hello world", "How are you?"],
)

for i, embedding in enumerate(response.data):
    print(f"Text {i}: {len(embedding.embedding)} dimensions")
```

### List Models

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",
)

models = client.models.list()
for model in models.data:
    print(model.id)
```

## Using httpx for RAG

The RAG endpoints are llmstack-specific (not part of the OpenAI API), so you need a regular HTTP client:

```bash
pip install httpx
```

### Ingest Documents

```python
import httpx

GATEWAY = "http://localhost:8000"
HEADERS = {
    "Authorization": "Bearer YOUR_KEY",
    "Content-Type": "application/json",
}

# Ingest a text document
response = httpx.post(
    f"{GATEWAY}/v1/rag/ingest",
    json={
        "text": open("whitepaper.txt").read(),
        "source": "whitepaper.txt",
    },
    headers=HEADERS,
    timeout=60,
)

print(response.json())
# {"status": "ok", "chunks": 42, "source": "whitepaper.txt"}
```

### Query with RAG

```python
import httpx

GATEWAY = "http://localhost:8000"
HEADERS = {
    "Authorization": "Bearer YOUR_KEY",
    "Content-Type": "application/json",
}

response = httpx.post(
    f"{GATEWAY}/v1/rag/query",
    json={
        "question": "What are the key findings?",
        "top_k": 5,
    },
    headers=HEADERS,
    timeout=120,
)

data = response.json()
print("Answer:", data["answer"])
print("Sources:", data["sources"])
```

### Streaming RAG Queries

```python
import httpx
import json

GATEWAY = "http://localhost:8000"
HEADERS = {
    "Authorization": "Bearer YOUR_KEY",
    "Content-Type": "application/json",
}

with httpx.stream(
    "POST",
    f"{GATEWAY}/v1/rag/query",
    json={
        "question": "Summarize the document.",
        "top_k": 5,
        "stream": True,
    },
    headers=HEADERS,
    timeout=httpx.Timeout(300, connect=10),
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            token = chunk.get("token", "")
            if token:
                print(token, end="", flush=True)
    print()
```

### Delete Documents

```python
import httpx

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer YOUR_KEY"}

response = httpx.delete(
    f"{GATEWAY}/v1/rag/documents/whitepaper.txt",
    headers=HEADERS,
)

print(response.json())
```

### Check RAG Status

```python
import httpx

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer YOUR_KEY"}

response = httpx.get(
    f"{GATEWAY}/v1/rag/status",
    headers=HEADERS,
)

print(response.json())
# {"collection": "llmstack", "points": 1234, "vectors_size": 1024}
```

## Using LangChain

llmstack is compatible with LangChain's ChatOpenAI class:

```bash
pip install langchain-openai
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",
    model="llama3.2",
)

response = llm.invoke("What is the meaning of life?")
print(response.content)
```

### LangChain with Streaming

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",
    model="llama3.2",
    streaming=True,
)

for chunk in llm.stream("Write a short poem about AI."):
    print(chunk.content, end="", flush=True)
print()
```

### LangChain Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",
    model="bge-m3",
)

vectors = embeddings.embed_documents(["Hello world", "How are you?"])
print(f"Generated {len(vectors)} embeddings of {len(vectors[0])} dimensions")
```

## Using LlamaIndex

```bash
pip install llama-index-llms-openai-like llama-index-embeddings-openai
```

```python
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding

llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key="YOUR_KEY",
    model="llama3.2",
    is_chat_model=True,
)

embed_model = OpenAIEmbedding(
    api_base="http://localhost:8000/v1",
    api_key="YOUR_KEY",
    model_name="bge-m3",
)

response = llm.complete("Explain transformers in ML.")
print(response.text)
```

## Error Handling

### Rate Limiting

When you exceed the rate limit, the API returns `429 Too Many Requests`. Handle it gracefully:

```python
import httpx
import time

def chat_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        response = client.post(
            "http://localhost:8000/v1/chat/completions",
            json={"model": "llama3.2", "messages": messages},
            headers={"Authorization": "Bearer YOUR_KEY"},
        )
        if response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 5))
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        return response.json()
    raise Exception("Rate limit exceeded after retries")
```

### Circuit Breaker

When the inference backend is down, the API returns `503 Service Unavailable`. This is the circuit breaker protecting the system:

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/chat/completions",
    json={"model": "llama3.2", "messages": [{"role": "user", "content": "Hi"}]},
    headers={"Authorization": "Bearer YOUR_KEY"},
)

if response.status_code == 503:
    print("Inference backend is temporarily unavailable. Try again later.")
```
