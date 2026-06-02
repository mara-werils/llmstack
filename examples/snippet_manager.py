"""Example: Using the snippet manager to save and search code snippets.

Usage:
    python examples/snippet_manager.py
"""

from llmstack.snippets.manager import SnippetManager


def main():
    # Initialize (uses ~/.llmstack/data/snippets.db by default)
    mgr = SnippetManager()

    # Save some snippets
    print("Saving snippets...")

    s1 = mgr.save(
        title="FastAPI Basic Route",
        code='''from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}''',
        language="python",
        tags=["fastapi", "web", "api"],
        description="Basic FastAPI hello world route",
    )
    print(f"  Saved: {s1.title} (id: {s1.id})")

    s2 = mgr.save(
        title="React useState Hook",
        code='''import { useState } from 'react';

function Counter() {
  const [count, setCount] = useState(0);
  return (
    <button onClick={() => setCount(count + 1)}>
      Count: {count}
    </button>
  );
}''',
        language="javascript",
        tags=["react", "hooks", "frontend"],
        description="Basic React counter with useState",
    )
    print(f"  Saved: {s2.title} (id: {s2.id})")

    # Search
    print("\nSearching for 'fastapi'...")
    results = mgr.search("fastapi")
    for s in results:
        print(f"  Found: {s.title} [{s.language}]")

    # List tags
    print("\nTags:")
    for tag, count in mgr.list_tags().items():
        print(f"  {tag}: {count}")

    # Stats
    print(f"\nTotal snippets: {mgr.count()}")
    print(f"Languages: {mgr.list_languages()}")


if __name__ == "__main__":
    main()
