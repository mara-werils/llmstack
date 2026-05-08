"""llmstack ask — ask questions about local files using a local LLM."""

from llmstack.ask.parsers import TextChunk, parse_file, collect_files
from llmstack.ask.embeddings import LocalEmbeddings
from llmstack.ask.engine import AskEngine, AskResult, SourceRef

__all__ = [
    "TextChunk",
    "parse_file",
    "collect_files",
    "LocalEmbeddings",
    "AskEngine",
    "AskResult",
    "SourceRef",
]
