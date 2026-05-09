"""Data preparation — convert various formats to chat training JSONL.

Supports: CSV, JSON, JSONL, TXT, Parquet, and raw conversation files.
Auto-detects format and column mapping.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset preparation."""

    input_column: str = ""      # auto-detected if empty
    output_column: str = ""     # auto-detected if empty
    system_prompt: str = ""     # optional system prompt for all examples
    max_samples: int = 0        # 0 = no limit
    eval_split: float = 0.1     # fraction for evaluation
    seed: int = 42
    min_length: int = 5         # skip examples shorter than this (chars)
    max_length: int = 0         # 0 = no limit


@dataclass
class DatasetStats:
    """Statistics about a prepared dataset."""

    total_examples: int = 0
    train_examples: int = 0
    eval_examples: int = 0
    skipped: int = 0
    avg_input_tokens: int = 0
    avg_output_tokens: int = 0
    total_tokens: int = 0
    source_format: str = ""

    def to_dict(self) -> dict:
        return {
            "total_examples": self.total_examples,
            "train_examples": self.train_examples,
            "eval_examples": self.eval_examples,
            "skipped": self.skipped,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "total_tokens": self.total_tokens,
            "source_format": self.source_format,
        }


@dataclass
class ChatExample:
    """A single training example in chat format."""

    messages: list[dict[str, str]]  # [{"role": "user", "content": "..."}, ...]

    def to_dict(self) -> dict:
        return {"messages": self.messages}

    def token_estimate(self) -> int:
        """Rough token count (~4 chars per token)."""
        total_chars = sum(len(m.get("content", "")) for m in self.messages)
        return max(1, total_chars // 4)


def detect_format(path: Path) -> str:
    """Detect the data file format."""
    suffix = path.suffix.lower()
    fmt_map = {
        ".jsonl": "jsonl",
        ".json": "json",
        ".csv": "csv",
        ".tsv": "csv",
        ".txt": "text",
        ".parquet": "parquet",
        ".md": "text",
    }
    return fmt_map.get(suffix, "unknown")


def _detect_columns(sample: dict) -> tuple[str, str]:
    """Auto-detect input/output column names from a sample row."""
    keys = set(sample.keys())

    # Chat format already
    if "messages" in keys:
        return "messages", ""

    # Common patterns
    input_candidates = [
        "input", "instruction", "prompt", "question", "query",
        "text", "human", "user", "source",
    ]
    output_candidates = [
        "output", "response", "answer", "completion", "reply",
        "assistant", "target", "label",
    ]

    input_col = ""
    output_col = ""

    for c in input_candidates:
        if c in keys:
            input_col = c
            break

    for c in output_candidates:
        if c in keys:
            output_col = c
            break

    # Fallback: first two string columns
    if not input_col:
        str_cols = [k for k, v in sample.items() if isinstance(v, str)]
        if len(str_cols) >= 2:
            input_col, output_col = str_cols[0], str_cols[1]
        elif len(str_cols) == 1:
            input_col = str_cols[0]

    return input_col, output_col


def _row_to_chat(
    row: dict, input_col: str, output_col: str, system_prompt: str,
) -> ChatExample | None:
    """Convert a data row to a ChatExample."""

    # Already in chat format
    if "messages" in row:
        msgs = row["messages"]
        if isinstance(msgs, str):
            try:
                msgs = json.loads(msgs)
            except json.JSONDecodeError:
                return None
        if isinstance(msgs, list) and msgs:
            return ChatExample(messages=msgs)
        return None

    if not input_col or input_col not in row:
        return None

    user_text = str(row[input_col]).strip()
    if not user_text:
        return None

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_text})

    if output_col and output_col in row:
        assistant_text = str(row[output_col]).strip()
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})

    return ChatExample(messages=messages)


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Try common wrapper keys
        for key in ("data", "examples", "rows", "train", "samples"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]
    return []


def _load_csv(path: Path) -> list[dict]:
    rows = []
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rows.append(dict(row))
    return rows


def _load_text(path: Path) -> list[dict]:
    """Load a plain text file as Q&A pairs separated by blank lines."""
    text = path.read_text(encoding="utf-8")
    blocks = text.split("\n\n")

    rows = []
    for i in range(0, len(blocks) - 1, 2):
        q = blocks[i].strip()
        a = blocks[i + 1].strip() if i + 1 < len(blocks) else ""
        if q:
            rows.append({"input": q, "output": a})

    # Fallback: each line as a separate input
    if not rows:
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append({"input": line, "output": ""})

    return rows


def _load_parquet(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(path)
        return table.to_pylist()
    except ImportError:
        logger.warning("pyarrow not installed — cannot read parquet files")
        return []


def load_raw_data(path: Path) -> tuple[list[dict], str]:
    """Load data from a file, returning (rows, format)."""
    fmt = detect_format(path)
    loaders = {
        "jsonl": _load_jsonl,
        "json": _load_json,
        "csv": _load_csv,
        "text": _load_text,
        "parquet": _load_parquet,
    }

    loader = loaders.get(fmt)
    if loader is None:
        raise ValueError(f"Unsupported file format: {path.suffix} ({fmt})")

    return loader(path), fmt


def prepare_dataset(
    path: Path,
    config: DatasetConfig | None = None,
    output_dir: Path | None = None,
) -> tuple[list[ChatExample], list[ChatExample], DatasetStats]:
    """Prepare a dataset for fine-tuning.

    Returns (train_examples, eval_examples, stats).
    Optionally writes train.jsonl and eval.jsonl to output_dir.
    """
    if config is None:
        config = DatasetConfig()

    rows, fmt = load_raw_data(path)
    if not rows:
        return [], [], DatasetStats(source_format=fmt)

    # Auto-detect columns
    input_col = config.input_column
    output_col = config.output_column
    if not input_col:
        input_col, output_col = _detect_columns(rows[0])
        logger.info("Auto-detected columns: input=%s, output=%s", input_col, output_col)

    # Convert to chat format
    examples: list[ChatExample] = []
    skipped = 0
    for row in rows:
        ex = _row_to_chat(row, input_col, output_col, config.system_prompt)
        if ex is None:
            skipped += 1
            continue

        # Length filters
        total_len = sum(len(m.get("content", "")) for m in ex.messages)
        if total_len < config.min_length:
            skipped += 1
            continue
        if config.max_length > 0 and total_len > config.max_length:
            skipped += 1
            continue

        examples.append(ex)

    if config.max_samples > 0 and len(examples) > config.max_samples:
        random.seed(config.seed)
        examples = random.sample(examples, config.max_samples)

    # Split train/eval
    random.seed(config.seed)
    random.shuffle(examples)
    eval_count = max(1, int(len(examples) * config.eval_split)) if examples else 0
    eval_examples = examples[:eval_count]
    train_examples = examples[eval_count:]

    # Compute stats
    input_tokens = []
    output_tokens = []
    for ex in examples:
        user_chars = sum(len(m["content"]) for m in ex.messages if m["role"] == "user")
        asst_chars = sum(len(m["content"]) for m in ex.messages if m["role"] == "assistant")
        input_tokens.append(max(1, user_chars // 4))
        output_tokens.append(max(1, asst_chars // 4))

    total_tokens = sum(input_tokens) + sum(output_tokens)

    stats = DatasetStats(
        total_examples=len(examples),
        train_examples=len(train_examples),
        eval_examples=len(eval_examples),
        skipped=skipped,
        avg_input_tokens=sum(input_tokens) // max(len(input_tokens), 1),
        avg_output_tokens=sum(output_tokens) // max(len(output_tokens), 1),
        total_tokens=total_tokens,
        source_format=fmt,
    )

    # Write output files
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "train.jsonl", train_examples)
        _write_jsonl(output_dir / "eval.jsonl", eval_examples)
        logger.info("Wrote %d train + %d eval examples to %s", len(train_examples), len(eval_examples), output_dir)

    return train_examples, eval_examples, stats


def _write_jsonl(path: Path, examples: list[ChatExample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
