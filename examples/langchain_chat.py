"""
LangChain integration with llmstack.

Demonstrates how to use LangChain with llmstack's OpenAI-compatible API
for chat, chains, streaming, and RAG with Qdrant vector search.

Install:
    pip install langchain langchain-openai langchain-qdrant qdrant-client

Usage:
    1. Start llmstack with RAG preset:  llmstack init --preset rag && llmstack up
    2. Run this script:                  python langchain_chat.py
"""

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ── Configuration ─────────────────────────────────────────────────────
LLMSTACK_URL = "http://localhost:8000/v1"
LLMSTACK_API_KEY = "llmstack"  # any non-empty string works
QDRANT_URL = "http://localhost:6333"


# ── 1. Simple chat completion ────────────────────────────────────────
def simple_chat():
    """Basic question-answer using ChatOpenAI pointed at llmstack."""
    llm = ChatOpenAI(
        base_url=LLMSTACK_URL,
        api_key=LLMSTACK_API_KEY,
        model="llama3.2",
        temperature=0.3,
        max_tokens=256,
    )

    response = llm.invoke("What are the three laws of thermodynamics?")
    print("[Simple chat]")
    print(response.content)
    print()


# ── 2. Prompt template + chain ───────────────────────────────────────
def chain_example():
    """Build an LCEL chain with a prompt template and output parser."""
    llm = ChatOpenAI(
        base_url=LLMSTACK_URL,
        api_key=LLMSTACK_API_KEY,
        model="llama3.2",
        temperature=0.4,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert {domain} teacher. Explain concepts simply."),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "domain": "physics",
        "question": "Why is the sky blue?",
    })
    print("[Chain]")
    print(result)
    print()


# ── 3. Streaming output ──────────────────────────────────────────────
def streaming_example():
    """Stream tokens to stdout as they are generated."""
    llm = ChatOpenAI(
        base_url=LLMSTACK_URL,
        api_key=LLMSTACK_API_KEY,
        model="llama3.2",
        temperature=0.5,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )

    print("[Streaming]")
    llm.invoke("Write a haiku about open-source software.")
    print("\n")


# ── 4. Batch processing ──────────────────────────────────────────────
def batch_example():
    """Process multiple prompts efficiently with .batch()."""
    llm = ChatOpenAI(
        base_url=LLMSTACK_URL,
        api_key=LLMSTACK_API_KEY,
        model="llama3.2",
        temperature=0.2,
        max_tokens=100,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Translate the following text to {language}. Reply with only the translation."),
        ("human", "{text}"),
    ])
    chain = prompt | llm | StrOutputParser()

    results = chain.batch([
        {"language": "French", "text": "Hello, how are you?"},
        {"language": "Spanish", "text": "The weather is beautiful today."},
        {"language": "Japanese", "text": "Open source is amazing."},
    ])

    print("[Batch translations]")
    for r in results:
        print(f"  {r}")
    print()


# ── 5. RAG with LangChain + llmstack embeddings + Qdrant ─────────────
def rag_example():
    """Full RAG pipeline: embed documents into Qdrant, then query.

    This connects directly to the Qdrant instance that llmstack manages,
    using llmstack's embeddings endpoint for vector generation.
    """
    # Use llmstack's OpenAI-compatible embeddings endpoint
    embeddings = OpenAIEmbeddings(
        base_url=LLMSTACK_URL,
        api_key=LLMSTACK_API_KEY,
        model="bge-m3",
    )

    # Connect to the Qdrant instance that llmstack started
    qdrant_client = QdrantClient(url=QDRANT_URL)
    collection_name = "langchain_demo"

    # Sample documents to index
    documents = [
        Document(
            page_content=(
                "LLMStack is an open-source tool that boots a full LLM inference stack "
                "with one command. It includes Ollama or vLLM for inference, Qdrant for "
                "vector search, Redis for caching, and a FastAPI gateway."
            ),
            metadata={"source": "llmstack-docs", "section": "overview"},
        ),
        Document(
            page_content=(
                "The llmstack gateway provides OpenAI-compatible endpoints for chat "
                "completions, embeddings, and model listing. It also includes built-in "
                "RAG endpoints for document ingestion and querying."
            ),
            metadata={"source": "llmstack-docs", "section": "gateway"},
        ),
        Document(
            page_content=(
                "llmstack supports automatic hardware detection. On NVIDIA GPUs with "
                "16GB+ VRAM it uses vLLM for maximum throughput. On Apple Silicon or "
                "lower VRAM it uses Ollama with Metal acceleration or GGUF quantization."
            ),
            metadata={"source": "llmstack-docs", "section": "hardware"},
        ),
        Document(
            page_content=(
                "The circuit breaker in llmstack prevents cascading failures when the "
                "inference backend goes down. After 5 consecutive failures it opens the "
                "circuit and returns 503 immediately, with exponential backoff on recovery."
            ),
            metadata={"source": "llmstack-docs", "section": "resilience"},
        ),
        Document(
            page_content=(
                "Redis is used for two purposes in llmstack: semantic response caching "
                "(SHA-256 hash of model + messages) and token-bucket rate limiting "
                "(atomic Lua scripts for race-free counting)."
            ),
            metadata={"source": "llmstack-docs", "section": "caching"},
        ),
    ]

    # Create vector store and add documents
    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=collection_name,
        force_recreate=True,  # start fresh for the demo
    )

    # Build a retrieval chain
    llm = ChatOpenAI(
        base_url=LLMSTACK_URL,
        api_key=LLMSTACK_API_KEY,
        model="llama3.2",
        temperature=0.1,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    rag_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Answer the question using only the provided context. "
            "If the context doesn't contain the answer, say so.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(
            f"[{d.metadata.get('section', 'unknown')}] {d.page_content}"
            for d in docs
        )

    # LCEL RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x,
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What backends does llmstack support for inference?",
        "How does the circuit breaker work?",
        "What is Redis used for in llmstack?",
    ]

    print("[RAG with LangChain + Qdrant]")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

    # Cleanup
    qdrant_client.delete_collection(collection_name)


# ── Run all examples ─────────────────────────────────────────────────
if __name__ == "__main__":
    simple_chat()
    chain_example()
    streaming_example()
    batch_example()
    rag_example()
