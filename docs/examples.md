# Examples

This page shows common integration patterns for llmstack across different frameworks and use cases.

## Prove the value (runnable)

Two self-contained scripts in [`examples/`](https://github.com/mara-werils/llmstack/tree/main/examples)
turn the "saves money" and "faster/private" claims into numbers you can reproduce:

```bash
python examples/savings_demo.py      # value local usage in dollars (savings engine)
python examples/benchmark_proof.py   # reproducible benchmark + zero-egress proof (exits non-zero on egress)
```

`savings_demo.py` runs the same calculator and ledger behind `llmstack savings`
against an isolated temp ledger. `benchmark_proof.py` runs the deterministic
benchmark suite under the egress monitor — the exact check the `Benchmark` CI
workflow gates on. See the [savings](guide/savings.md) and
[benchmarks](guide/benchmarks.md) guides.

## Python (openai SDK)

The simplest way to use llmstack from Python is with the official OpenAI client:

```bash
pip install openai
```

### Basic Chat

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is machine learning?"}],
)
print(response.choices[0].message.content)
```

### Multi-Turn Conversation

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_KEY")

messages = [
    {"role": "system", "content": "You are a Python tutor. Give short, clear answers."},
]

questions = [
    "What is a list comprehension?",
    "Show me an example.",
    "How does it compare to a for loop?",
]

for question in questions:
    messages.append({"role": "user", "content": question})
    response = client.chat.completions.create(model="llama3.2", messages=messages)
    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Streaming with Progress

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="YOUR_KEY")

stream = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Write a short story about a robot."}],
    stream=True,
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

## RAG Pipeline

Build a document Q&A system using llmstack's built-in RAG endpoints:

```python
import httpx

GATEWAY = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer YOUR_KEY", "Content-Type": "application/json"}


def ingest_file(filepath: str) -> dict:
    """Ingest a text file into the RAG pipeline."""
    text = open(filepath).read()
    response = httpx.post(
        f"{GATEWAY}/v1/rag/ingest",
        json={"text": text, "source": filepath},
        headers=HEADERS,
        timeout=120,
    )
    return response.json()


def ask(question: str, top_k: int = 5) -> dict:
    """Ask a question using RAG."""
    response = httpx.post(
        f"{GATEWAY}/v1/rag/query",
        json={"question": question, "top_k": top_k},
        headers=HEADERS,
        timeout=120,
    )
    return response.json()


# Ingest documents
for doc in ["whitepaper.txt", "faq.txt", "guide.txt"]:
    result = ingest_file(doc)
    print(f"Ingested {doc}: {result['chunks']} chunks")

# Ask questions
result = ask("What are the key findings of the whitepaper?")
print(f"\nAnswer: {result['answer']}")
print(f"Sources: {[s['source'] for s in result['sources']]}")
```

## LangChain

### Chat with Memory

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",
    model="llama3.2",
)

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)

print(chain.predict(input="Hi! I'm working on a Python project."))
print(chain.predict(input="What testing framework do you recommend?"))
print(chain.predict(input="How do I set it up?"))
```

### Embeddings for Similarity Search

```python
from langchain_openai import OpenAIEmbeddings
import numpy as np

embeddings = OpenAIEmbeddings(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_KEY",
    model="bge-m3",
)

texts = [
    "Python is a programming language",
    "JavaScript is used for web development",
    "Machine learning uses statistical models",
    "Docker containers package applications",
]

vectors = embeddings.embed_documents(texts)
query_vector = embeddings.embed_query("What language is used for AI?")

# Compute cosine similarity
similarities = []
for i, vec in enumerate(vectors):
    similarity = np.dot(query_vector, vec) / (np.linalg.norm(query_vector) * np.linalg.norm(vec))
    similarities.append((texts[i], similarity))

for text, score in sorted(similarities, key=lambda x: x[1], reverse=True):
    print(f"{score:.3f} | {text}")
```

## LlamaIndex

### Simple Query Engine

```python
from llama_index.core import VectorStoreIndex, Document
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

Settings.llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key="YOUR_KEY",
    model="llama3.2",
    is_chat_model=True,
)

Settings.embed_model = OpenAIEmbedding(
    api_base="http://localhost:8000/v1",
    api_key="YOUR_KEY",
    model_name="bge-m3",
)

documents = [
    Document(text="llmstack boots a full LLM stack with one command."),
    Document(text="It includes inference, embeddings, vector DB, and caching."),
    Document(text="The gateway provides an OpenAI-compatible API."),
]

index = VectorStoreIndex.from_documents(documents)
engine = index.as_query_engine()

response = engine.query("What does llmstack include?")
print(response)
```

## cURL

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain REST APIs in one paragraph."}
    ]
  }'
```

### Streaming Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Count from 1 to 10."}],
    "stream": true
  }'
```

### Generate Embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": ["Hello world"]
  }'
```

### Check Health

```bash
curl http://localhost:8000/healthz | python -m json.tool
```

## TypeScript / Node.js

Using the official OpenAI Node.js SDK:

```bash
npm install openai
```

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "YOUR_KEY",
});

async function main() {
  const response = await client.chat.completions.create({
    model: "llama3.2",
    messages: [{ role: "user", content: "What is TypeScript?" }],
  });

  console.log(response.choices[0].message.content);
}

main();
```

### Streaming in Node.js

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "YOUR_KEY",
});

async function main() {
  const stream = await client.chat.completions.create({
    model: "llama3.2",
    messages: [{ role: "user", content: "Write a haiku about coding." }],
    stream: true,
  });

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      process.stdout.write(content);
    }
  }
  console.log();
}

main();
```

## Vercel AI SDK

```typescript
import { generateText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";

const llmstack = createOpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "YOUR_KEY",
});

const { text } = await generateText({
  model: llmstack("llama3.2"),
  prompt: "What is the meaning of life?",
});

console.log(text);
```

## Docker Compose (Exported)

Share your stack with teammates who do not have llmstack installed:

```bash
# Generate a standalone docker-compose.yml
llmstack export

# Share the file, then recipients run:
docker compose up -d
```

The exported file includes all services, volumes, networks, and environment variables. No llmstack dependency required.
