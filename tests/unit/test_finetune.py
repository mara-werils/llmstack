"""Comprehensive tests for the fine-tuning pipeline.

Covers data preparation, format detection, column auto-detection,
hyperparameter selection, model size estimation, training config,
eval scoring, export config, and config schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmstack.finetune.data import (
    ChatExample,
    DatasetConfig,
    DatasetStats,
    _detect_columns,
    _load_parquet,
    _row_to_chat,
    _write_jsonl,
    detect_format,
    load_raw_data,
    prepare_dataset,
)
from llmstack.finetune.hyperparams import (
    auto_hyperparams,
    estimate_model_size,
)
from llmstack.finetune.trainer import TrainConfig, TrainResult
from llmstack.finetune.eval import _word_overlap, _length_ratio, _combined_score, EvalResult
from llmstack.finetune.export import ExportResult


# ===================================================================
# ChatExample tests
# ===================================================================


class TestChatExample:
    def test_to_dict(self):
        ex = ChatExample(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        d = ex.to_dict()
        assert d["messages"][0]["role"] == "user"
        assert d["messages"][1]["content"] == "hi"

    def test_token_estimate(self):
        ex = ChatExample(
            messages=[
                {"role": "user", "content": "x" * 400},
            ]
        )
        tokens = ex.token_estimate()
        assert tokens == 100  # 400 chars / 4

    def test_empty_message_token_estimate(self):
        ex = ChatExample(messages=[{"role": "user", "content": ""}])
        assert ex.token_estimate() >= 1


# ===================================================================
# Format detection tests
# ===================================================================


class TestFormatDetection:
    def test_jsonl(self):
        assert detect_format(Path("data.jsonl")) == "jsonl"

    def test_json(self):
        assert detect_format(Path("data.json")) == "json"

    def test_csv(self):
        assert detect_format(Path("data.csv")) == "csv"

    def test_tsv(self):
        assert detect_format(Path("data.tsv")) == "csv"

    def test_txt(self):
        assert detect_format(Path("data.txt")) == "text"

    def test_parquet(self):
        assert detect_format(Path("data.parquet")) == "parquet"

    def test_md(self):
        assert detect_format(Path("data.md")) == "text"

    def test_unknown(self):
        assert detect_format(Path("data.xyz")) == "unknown"


# ===================================================================
# Column detection tests
# ===================================================================


class TestColumnDetection:
    def test_instruction_output(self):
        i, o = _detect_columns({"instruction": "hi", "output": "hello"})
        assert i == "instruction"
        assert o == "output"

    def test_input_response(self):
        i, o = _detect_columns({"input": "hi", "response": "hello"})
        assert i == "input"
        assert o == "response"

    def test_prompt_completion(self):
        i, o = _detect_columns({"prompt": "hi", "completion": "hello"})
        assert i == "prompt"
        assert o == "completion"

    def test_question_answer(self):
        i, o = _detect_columns({"question": "hi", "answer": "hello"})
        assert i == "question"
        assert o == "answer"

    def test_messages_column(self):
        i, o = _detect_columns({"messages": [{"role": "user", "content": "hi"}]})
        assert i == "messages"
        assert o == ""

    def test_fallback_to_first_two(self):
        i, o = _detect_columns({"col_a": "hello", "col_b": "world"})
        assert i == "col_a"
        assert o == "col_b"

    def test_human_assistant(self):
        i, o = _detect_columns({"human": "hi", "assistant": "hello"})
        assert i == "human"
        assert o == "assistant"


# ===================================================================
# Row to chat conversion tests
# ===================================================================


class TestRowToChat:
    def test_basic_conversion(self):
        ex = _row_to_chat(
            {"input": "hello", "output": "hi there"},
            "input",
            "output",
            "",
        )
        assert ex is not None
        assert len(ex.messages) == 2
        assert ex.messages[0]["role"] == "user"
        assert ex.messages[1]["role"] == "assistant"

    def test_with_system_prompt(self):
        ex = _row_to_chat(
            {"input": "hello", "output": "hi"},
            "input",
            "output",
            "You are helpful.",
        )
        assert ex is not None
        assert len(ex.messages) == 3
        assert ex.messages[0]["role"] == "system"
        assert ex.messages[0]["content"] == "You are helpful."

    def test_empty_input_skipped(self):
        ex = _row_to_chat({"input": "", "output": "hi"}, "input", "output", "")
        assert ex is None

    def test_missing_column_skipped(self):
        ex = _row_to_chat({"other": "data"}, "input", "output", "")
        assert ex is None

    def test_messages_format_passthrough(self):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        ex = _row_to_chat({"messages": msgs}, "messages", "", "")
        assert ex is not None
        assert len(ex.messages) == 2

    def test_messages_as_json_string(self):
        msgs = json.dumps([{"role": "user", "content": "hi"}])
        ex = _row_to_chat({"messages": msgs}, "messages", "", "")
        assert ex is not None
        assert ex.messages[0]["content"] == "hi"

    def test_no_output_column(self):
        ex = _row_to_chat({"input": "hello"}, "input", "", "")
        assert ex is not None
        assert len(ex.messages) == 1
        assert ex.messages[0]["role"] == "user"


# ===================================================================
# Data loading tests
# ===================================================================


class TestLoadData:
    def test_load_jsonl(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"input": "q1", "output": "a1"}),
            json.dumps({"input": "q2", "output": "a2"}),
        ]
        f.write_text("\n".join(lines))

        rows, fmt = load_raw_data(f)
        assert fmt == "jsonl"
        assert len(rows) == 2
        assert rows[0]["input"] == "q1"

    def test_load_json_array(self, tmp_path):
        f = tmp_path / "data.json"
        data = [{"input": "q1", "output": "a1"}, {"input": "q2", "output": "a2"}]
        f.write_text(json.dumps(data))

        rows, fmt = load_raw_data(f)
        assert fmt == "json"
        assert len(rows) == 2

    def test_load_json_wrapped(self, tmp_path):
        f = tmp_path / "data.json"
        data = {"data": [{"input": "q1", "output": "a1"}]}
        f.write_text(json.dumps(data))

        rows, fmt = load_raw_data(f)
        assert len(rows) == 1

    def test_load_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("input,output\nhello,world\nfoo,bar\n")

        rows, fmt = load_raw_data(f)
        assert fmt == "csv"
        assert len(rows) == 2
        assert rows[0]["input"] == "hello"
        assert rows[0]["output"] == "world"

    def test_load_text(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text(
            "What is Python?\n\nPython is a programming language.\n\nWhat is Rust?\n\nRust is a systems language.\n"
        )

        rows, fmt = load_raw_data(f)
        assert fmt == "text"
        assert len(rows) >= 2

    def test_unsupported_format(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("stuff")

        with pytest.raises(ValueError, match="Unsupported"):
            load_raw_data(f)


# ===================================================================
# Dataset preparation tests
# ===================================================================


class TestPrepareDataset:
    def test_full_pipeline(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [json.dumps({"input": f"question {i}", "output": f"answer {i}"}) for i in range(20)]
        f.write_text("\n".join(lines))

        train, eval_, stats = prepare_dataset(f)
        assert stats.total_examples == 20
        assert stats.train_examples + stats.eval_examples == 20
        assert len(train) == stats.train_examples
        assert len(eval_) == stats.eval_examples

    def test_write_output_files(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [json.dumps({"input": f"q{i}", "output": f"a{i}"}) for i in range(10)]
        f.write_text("\n".join(lines))

        out = tmp_path / "output"
        prepare_dataset(f, output_dir=out)

        assert (out / "train.jsonl").exists()
        assert (out / "eval.jsonl").exists()

    def test_min_length_filter(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"input": "hi", "output": ""}),  # too short
            json.dumps({"input": "a longer question here", "output": "a longer answer here"}),
        ]
        f.write_text("\n".join(lines))

        config = DatasetConfig(min_length=10)
        train, eval_, stats = prepare_dataset(f, config=config)
        assert stats.skipped >= 1

    def test_max_samples(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [json.dumps({"input": f"q{i}", "output": f"a{i}"}) for i in range(50)]
        f.write_text("\n".join(lines))

        config = DatasetConfig(max_samples=10)
        train, eval_, stats = prepare_dataset(f, config=config)
        assert stats.total_examples == 10

    def test_system_prompt(self, tmp_path):
        f = tmp_path / "data.jsonl"
        # Need enough examples so train set is non-empty after split
        lines = [json.dumps({"input": f"hello {i}", "output": f"hi {i}"}) for i in range(10)]
        f.write_text("\n".join(lines))

        config = DatasetConfig(system_prompt="You are a bot.", eval_split=0.1)
        train, _, _ = prepare_dataset(f, config=config)
        assert len(train) > 0
        assert train[0].messages[0]["role"] == "system"
        assert train[0].messages[0]["content"] == "You are a bot."

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.jsonl"
        f.write_text("")

        train, eval_, stats = prepare_dataset(f)
        assert stats.total_examples == 0
        assert len(train) == 0

    def test_stats_format(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [json.dumps({"input": f"q{i}", "output": f"a{i}"}) for i in range(10)]
        f.write_text("\n".join(lines))

        _, _, stats = prepare_dataset(f)
        d = stats.to_dict()
        assert "total_examples" in d
        assert "source_format" in d
        assert d["source_format"] == "jsonl"

    def test_skips_rows_that_fail_conversion(self, tmp_path):
        f = tmp_path / "data.jsonl"
        rows = [{"input": f"a long enough question number {i}", "output": f"a long enough answer {i}"} for i in range(8)]
        rows.append({"other": "no usable columns here at all"})
        f.write_text("\n".join(json.dumps(r) for r in rows))

        train, eval_, stats = prepare_dataset(f)
        assert stats.skipped >= 1

    def test_max_length_filter(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps({"input": "a reasonably long question here", "output": "a reasonably long answer here"})
            for _ in range(8)
        ]
        f.write_text("\n".join(lines))

        config = DatasetConfig(max_length=10)
        train, eval_, stats = prepare_dataset(f, config=config)
        assert stats.skipped == 8

    def test_writes_non_empty_jsonl_files(self, tmp_path):
        f = tmp_path / "data.jsonl"
        lines = [
            json.dumps(
                {
                    "input": f"a long enough question number {i} for the min length filter",
                    "output": f"a long enough answer number {i} for the min length filter",
                }
            )
            for i in range(10)
        ]
        f.write_text("\n".join(lines))

        out = tmp_path / "output"
        train, eval_, stats = prepare_dataset(f, output_dir=out)

        assert len(train) > 0
        assert len(eval_) > 0
        train_lines = (out / "train.jsonl").read_text().strip().splitlines()
        eval_lines = (out / "eval.jsonl").read_text().strip().splitlines()
        assert len(train_lines) == len(train)
        assert len(eval_lines) == len(eval_)


# ===================================================================
# Column detection / row conversion edge cases
# ===================================================================


class TestColumnDetectionFallbackSingleColumn:
    def test_single_string_column_used_as_input(self):
        i, o = _detect_columns({"only_text": "hello", "count": 5})
        assert i == "only_text"
        assert o == ""


class TestRowToChatMessagesEdgeCases:
    def test_invalid_json_string_returns_none(self):
        ex = _row_to_chat({"messages": "{not valid json"}, "messages", "", "")
        assert ex is None

    def test_empty_messages_list_returns_none(self):
        ex = _row_to_chat({"messages": []}, "messages", "", "")
        assert ex is None

    def test_non_list_messages_returns_none(self):
        ex = _row_to_chat({"messages": {"role": "user"}}, "messages", "", "")
        assert ex is None


# ===================================================================
# Loader edge cases
# ===================================================================


class TestLoaderEdgeCases:
    def test_jsonl_skips_invalid_lines(self, tmp_path):
        f = tmp_path / "data.jsonl"
        f.write_text('{"input": "ok", "output": "fine"}\nnot valid json\n')
        rows, fmt = load_raw_data(f)
        assert len(rows) == 1

    def test_json_dict_without_known_wrapper_key(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"input": "single row", "output": "answer"}))
        rows, fmt = load_raw_data(f)
        assert rows == [{"input": "single row", "output": "answer"}]

    def test_json_scalar_returns_empty(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text(json.dumps("just a string"))
        rows, fmt = load_raw_data(f)
        assert rows == []

    def test_text_fallback_per_line(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("line one\nline two\nline three")
        rows, fmt = load_raw_data(f)
        assert fmt == "text"
        assert len(rows) == 3
        assert rows[0] == {"input": "line one", "output": ""}

    def test_parquet_without_pyarrow_returns_empty(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_bytes(b"fake parquet bytes")
        with patch.dict(sys.modules, {"pyarrow": None, "pyarrow.parquet": None}):
            rows, fmt = load_raw_data(f)
        assert rows == []
        assert fmt == "parquet"

    def test_parquet_with_pyarrow_installed(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_bytes(b"fake parquet bytes")

        fake_table = MagicMock()
        fake_table.to_pylist.return_value = [{"input": "q1", "output": "a1"}]
        fake_pq = MagicMock(read_table=MagicMock(return_value=fake_table))
        fake_pyarrow = MagicMock(parquet=fake_pq)

        with patch.dict(sys.modules, {"pyarrow": fake_pyarrow, "pyarrow.parquet": fake_pq}):
            rows = _load_parquet(f)

        assert rows == [{"input": "q1", "output": "a1"}]


class TestWriteJsonlDirectly:
    def test_writes_each_example_as_a_line(self, tmp_path):
        examples = [
            ChatExample(messages=[{"role": "user", "content": "hi"}]),
            ChatExample(messages=[{"role": "user", "content": "bye"}]),
        ]
        out = tmp_path / "out.jsonl"
        _write_jsonl(out, examples)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["messages"][0]["content"] == "hi"


# ===================================================================
# Hyperparameter auto-selection tests
# ===================================================================


class TestAutoHyperparams:
    def test_small_dataset(self):
        hp = auto_hyperparams(num_examples=50, model_size_b=7.0)
        assert hp.epochs == 5
        assert hp.learning_rate < 2e-4  # lower LR for small datasets

    def test_medium_dataset(self):
        hp = auto_hyperparams(num_examples=2000, model_size_b=7.0)
        assert hp.epochs == 2
        assert hp.lora_r == 16

    def test_large_dataset(self):
        hp = auto_hyperparams(num_examples=50000, model_size_b=7.0)
        assert hp.epochs == 1
        assert hp.lora_r >= 32  # higher rank for large datasets

    def test_large_model_smaller_batch(self):
        hp = auto_hyperparams(num_examples=1000, model_size_b=70.0)
        assert hp.batch_size <= 2
        assert hp.gradient_accumulation_steps >= 8

    def test_small_model_larger_batch(self):
        hp = auto_hyperparams(num_examples=1000, model_size_b=1.0)
        assert hp.batch_size >= 4

    def test_qlora_enables_4bit(self):
        hp = auto_hyperparams(num_examples=100, method="qlora")
        assert hp.use_4bit is True

    def test_lora_disables_4bit(self):
        hp = auto_hyperparams(num_examples=100, method="lora")
        assert hp.use_4bit is False

    def test_max_seq_length_preserved(self):
        hp = auto_hyperparams(num_examples=100, max_seq_length=4096)
        assert hp.max_seq_length == 4096

    def test_to_dict(self):
        hp = auto_hyperparams(num_examples=100)
        d = hp.to_dict()
        assert "lora_r" in d
        assert "epochs" in d
        assert "learning_rate" in d
        assert "use_4bit" in d

    def test_logging_steps_positive(self):
        hp = auto_hyperparams(num_examples=10)
        assert hp.logging_steps >= 1
        assert hp.save_steps >= 1
        assert hp.eval_steps >= 1


# ===================================================================
# Model size estimation tests
# ===================================================================


class TestEstimateModelSize:
    def test_llama_70b(self):
        assert estimate_model_size("llama3.1:70b") == 70.0

    def test_llama_8b(self):
        assert estimate_model_size("llama3.2:8b") == 8.0

    def test_llama_1b(self):
        assert estimate_model_size("llama3.2:1b") == 1.0

    def test_mistral_7b(self):
        assert estimate_model_size("mistral-7b-instruct") == 7.0

    def test_qwen_14b(self):
        assert estimate_model_size("qwen2.5-14b") == 14.0

    def test_default(self):
        assert estimate_model_size("some-custom-model") == 7.0

    def test_unsloth_format(self):
        assert estimate_model_size("unsloth/llama-3.2-1b-instruct-bnb-4bit") == 1.0


# ===================================================================
# TrainConfig and TrainResult tests
# ===================================================================


class TestTrainConfig:
    def test_defaults(self):
        config = TrainConfig()
        assert config.method == "qlora"
        assert config.output_dir == "./finetune-output"

    def test_custom_config(self):
        config = TrainConfig(
            base_model="custom/model",
            method="lora",
            output_dir="/tmp/output",
        )
        assert config.base_model == "custom/model"
        assert config.method == "lora"


class TestTrainResult:
    def test_to_dict(self):
        result = TrainResult(
            success=True,
            output_dir="/tmp/out",
            adapter_path="/tmp/out/adapter",
            final_loss=0.5,
            best_loss=0.3,
            total_steps=100,
            train_time_seconds=120.5,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["final_loss"] == 0.5
        assert d["best_loss"] == 0.3

    def test_error_result(self):
        result = TrainResult(success=False, error="GPU OOM")
        d = result.to_dict()
        assert d["success"] is False
        assert d["error"] == "GPU OOM"


# ===================================================================
# Eval scoring tests
# ===================================================================


class TestEvalScoring:
    def test_word_overlap_identical(self):
        assert _word_overlap("hello world", "hello world") == pytest.approx(1.0)

    def test_word_overlap_partial(self):
        score = _word_overlap("hello world foo", "hello world bar")
        assert 0.3 < score < 0.8

    def test_word_overlap_no_match(self):
        assert _word_overlap("hello", "goodbye") == pytest.approx(0.0)

    def test_word_overlap_empty(self):
        assert _word_overlap("", "hello") == 0.0
        assert _word_overlap("hello", "") == 0.0

    def test_length_ratio_same(self):
        assert _length_ratio("hello", "world") == pytest.approx(1.0)

    def test_length_ratio_too_short(self):
        score = _length_ratio("a long reference text here", "x")
        assert score < 0.3

    def test_length_ratio_too_long(self):
        score = _length_ratio("x", "a" * 1000)
        assert score < 0.5

    def test_combined_score(self):
        score = _combined_score("hello world", "hello world")
        assert score > 0.8

    def test_combined_score_different(self):
        score = _combined_score("hello world", "completely different text")
        assert score < 0.5


class TestEvalResult:
    def test_to_dict(self):
        result = EvalResult(
            base_model="llama3.2",
            tuned_model="my-model",
            num_examples=10,
            base_avg_score=0.4,
            tuned_avg_score=0.7,
            improvement_pct=75.0,
        )
        d = result.to_dict()
        assert d["improvement_pct"] == 75.0
        assert d["base_model"] == "llama3.2"


# ===================================================================
# Export tests
# ===================================================================


class TestExportResult:
    def test_to_dict(self):
        r = ExportResult(
            success=True,
            gguf_path="/tmp/model.gguf",
            ollama_model="my-model",
            size_mb=1234.5,
            quantization="q4_k_m",
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["quantization"] == "q4_k_m"
        assert d["size_mb"] == 1234.5

    def test_error_result(self):
        r = ExportResult(success=False, error="unsloth not installed")
        d = r.to_dict()
        assert d["error"] == "unsloth not installed"


# ===================================================================
# Config schema tests
# ===================================================================


class TestFinetuneConfigSchema:
    def test_defaults(self):
        from llmstack.config.schema import FinetuneConfig

        config = FinetuneConfig()
        assert config.method == "qlora"
        assert config.lora_r == 16
        assert config.eval_split == 0.1

    def test_custom(self):
        from llmstack.config.schema import FinetuneConfig

        config = FinetuneConfig(
            base_model="custom/model",
            method="lora",
            lora_r=32,
            epochs=5,
        )
        assert config.lora_r == 32
        assert config.epochs == 5

    def test_stack_config_has_finetune(self):
        from llmstack.config.schema import StackConfig

        config = StackConfig()
        assert hasattr(config, "finetune")
        assert config.finetune.method == "qlora"


# ===================================================================
# DatasetStats tests
# ===================================================================


class TestDatasetStats:
    def test_to_dict(self):
        stats = DatasetStats(
            total_examples=100,
            train_examples=90,
            eval_examples=10,
            avg_input_tokens=50,
            avg_output_tokens=30,
            total_tokens=8000,
            source_format="jsonl",
        )
        d = stats.to_dict()
        assert d["total_examples"] == 100
        assert d["source_format"] == "jsonl"
