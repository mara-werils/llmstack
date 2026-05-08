"""
OpenAI Python SDK with llmstack — drop-in replacement.

llmstack exposes a fully OpenAI-compatible API, so you can point the
official ``openai`` Python package at it with zero code changes beyond
the ``base_url``.

Install:
    pip install openai

Usage:
    1. Start llmstack:  llmstack up
    2. Run this script:  python openai_sdk.py
"""

from openai import OpenAI

# ── Connect to llmstack ──────────────────────────────────────────────
# The only change vs. OpenAI cloud: base_url points to your local gateway.
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="llmstack",  # any non-empty string; llmstack validates format, not value by default
)


# ── 1. List available models ─────────────────────────────────────────
def list_models():
    """Show every model the inference backend exposes."""
    models = client.models.list()
    print("Available models:")
    for m in models.data:
        print(f"  - {m.id}")
    print()


# ── 2. Simple chat completion ────────────────────────────────────────
def simple_chat():
    """One-shot question/answer with the default model."""
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.2,
        max_tokens=256,
    )
    print("[Chat completion]")
    print(f"Model: {response.model}")
    print(f"Answer: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
    print()


# ── 3. Multi-turn conversation ───────────────────────────────────────
def multi_turn_chat():
    """Maintain conversation history across multiple turns."""
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python function that reverses a string."},
    ]

    # First turn
    resp1 = client.chat.completions.create(
        model="llama3.2", messages=messages, temperature=0.3
    )
    assistant_reply = resp1.choices[0].message.content
    print("[Multi-turn — Turn 1]")
    print(assistant_reply)
    print()

    # Second turn — follow-up
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "Now add type hints and a docstring."})

    resp2 = client.chat.completions.create(
        model="llama3.2", messages=messages, temperature=0.3
    )
    print("[Multi-turn — Turn 2]")
    print(resp2.choices[0].message.content)
    print()


# ── 4. Streaming chat completion ─────────────────────────────────────
def streaming_chat():
    """Stream tokens as they arrive — great for real-time UIs."""
    print("[Streaming]")
    stream = client.chat.completions.create(
        model="llama3.2",
        messages=[{"role": "user", "content": "Explain quantum entanglement in 3 sentences."}],
        stream=True,
        temperature=0.4,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print("\n")


# ── 5. Embeddings ────────────────────────────────────────────────────
def create_embeddings():
    """Generate embeddings for semantic search, clustering, etc."""
    texts = [
        "LLMStack is an open-source LLM infrastructure tool.",
        "Quantum computing uses qubits instead of classical bits.",
        "The Eiffel Tower is located in Paris, France.",
    ]
    response = client.embeddings.create(model="bge-m3", input=texts)

    print("[Embeddings]")
    for i, emb in enumerate(response.data):
        vec = emb.embedding
        print(f"  Text {i}: dim={len(vec)}, first 5 values={vec[:5]}")
    print(f"Total tokens: {response.usage.total_tokens}")
    print()


# ── 6. JSON mode (structured output) ─────────────────────────────────
def json_mode():
    """Request structured JSON output from the model."""
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that responds in JSON format.",
            },
            {
                "role": "user",
                "content": (
                    "List the top 3 programming languages for machine learning. "
                    "Return JSON with a 'languages' array, each item having 'name' and 'reason' fields."
                ),
            },
        ],
        temperature=0.1,
        max_tokens=512,
    )
    print("[JSON mode]")
    print(response.choices[0].message.content)
    print()


# ── Run all examples ─────────────────────────────────────────────────
if __name__ == "__main__":
    list_models()
    simple_chat()
    multi_turn_chat()
    streaming_chat()
    create_embeddings()
    json_mode()
