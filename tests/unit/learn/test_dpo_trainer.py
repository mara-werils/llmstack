"""Tests for the DPO trainer (Direct Preference Optimization)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmstack.learn import dpo_trainer as dpo_mod
from llmstack.learn.dataset import DPOExample
from llmstack.learn.dpo_trainer import (
    DPOConfig,
    DPOTrainer,
    DPOTrainResult,
    _check_dpo_dependencies,
    prepare_dpo_dataset,
)


@pytest.fixture
def examples():
    return [
        DPOExample(
            prompt="What is 2+2?",
            chosen="The answer is 4.",
            rejected="I don't know.",
        ),
        DPOExample(
            prompt="Capital of France?",
            chosen="Paris is the capital of France.",
            rejected="London.",
        ),
    ]


class TestDPOConfig:
    def test_defaults(self):
        cfg = DPOConfig()
        assert cfg.base_model == "unsloth/llama-3.2-1b-instruct-bnb-4bit"
        assert cfg.beta == 0.1
        assert cfg.learning_rate == 5e-6
        assert cfg.epochs == 1
        assert cfg.batch_size == 2
        assert cfg.max_length == 1024
        assert cfg.max_prompt_length == 512
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.warmup_ratio == 0.1
        assert cfg.lora_r == 16
        assert cfg.lora_alpha == 32
        assert cfg.use_4bit is True
        # output_dir defaults under the user's home training dir
        assert cfg.output_dir.endswith(str(Path(".llmstack") / "training" / "dpo"))

    def test_overrides(self):
        cfg = DPOConfig(base_model="my-model", beta=0.5, epochs=3, use_4bit=False)
        assert cfg.base_model == "my-model"
        assert cfg.beta == 0.5
        assert cfg.epochs == 3
        assert cfg.use_4bit is False


class TestDPOTrainResult:
    def test_defaults(self):
        result = DPOTrainResult()
        assert result.success is False
        assert result.output_dir == ""
        assert result.adapter_path == ""
        assert result.final_loss == 0.0
        assert result.error is None

    def test_to_dict_rounds_values(self):
        result = DPOTrainResult(
            success=True,
            output_dir="/tmp/out",
            adapter_path="/tmp/out/adapter",
            final_loss=0.123456,
            rewards_chosen=1.987654,
            rewards_rejected=0.111111,
            reward_margin=1.876543,
            train_time_seconds=12.3456,
            total_steps=42,
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output_dir"] == "/tmp/out"
        assert d["adapter_path"] == "/tmp/out/adapter"
        assert d["final_loss"] == 0.1235
        assert d["rewards_chosen"] == 1.9877
        assert d["rewards_rejected"] == 0.1111
        assert d["reward_margin"] == 1.8765
        assert d["train_time_seconds"] == 12.3
        assert d["total_steps"] == 42
        assert d["error"] is None

    def test_to_dict_keys(self):
        d = DPOTrainResult().to_dict()
        expected = {
            "success",
            "output_dir",
            "adapter_path",
            "final_loss",
            "rewards_chosen",
            "rewards_rejected",
            "reward_margin",
            "train_time_seconds",
            "total_steps",
            "error",
        }
        assert set(d.keys()) == expected

    def test_to_dict_with_error(self):
        d = DPOTrainResult(success=False, error="boom").to_dict()
        assert d["success"] is False
        assert d["error"] == "boom"

    def test_to_dict_is_json_serializable(self):
        json.dumps(DPOTrainResult().to_dict())


class TestCheckDependencies:
    def test_returns_tuple(self):
        has_deps, backend = _check_dpo_dependencies()
        assert isinstance(has_deps, bool)
        assert isinstance(backend, str)

    def test_missing_dependencies(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in {"trl", "transformers", "peft"}:
                raise ImportError(f"No module named {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        has_deps, backend = _check_dpo_dependencies()
        assert has_deps is False
        assert backend == "missing"

    def test_present_dependencies(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name in {"trl", "transformers", "peft"}:
                return type(name, (), {})()
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        has_deps, backend = _check_dpo_dependencies()
        assert has_deps is True
        assert backend == "trl + peft"


class TestDPOTrainerInit:
    def test_default_config(self):
        trainer = DPOTrainer()
        assert isinstance(trainer.config, DPOConfig)
        assert trainer._progress_callback is None

    def test_custom_config(self):
        cfg = DPOConfig(base_model="custom")
        trainer = DPOTrainer(config=cfg)
        assert trainer.config is cfg

    def test_set_progress_callback(self):
        trainer = DPOTrainer()
        calls = []

        def cb(step, total, loss):
            calls.append((step, total, loss))

        trainer.set_progress_callback(cb)
        assert trainer._progress_callback is cb
        trainer._progress_callback(1, 10, 0.5)
        assert calls == [(1, 10, 0.5)]


class TestTrain:
    def test_missing_dependencies_returns_error(self, monkeypatch, examples):
        monkeypatch.setattr(dpo_mod, "_check_dpo_dependencies", lambda: (False, "missing"))
        trainer = DPOTrainer()
        result = trainer.train(examples)
        assert result.success is False
        assert result.error is not None
        assert "pip install trl peft transformers" in result.error

    def test_empty_examples_returns_error(self, monkeypatch):
        monkeypatch.setattr(dpo_mod, "_check_dpo_dependencies", lambda: (True, "trl + peft"))
        trainer = DPOTrainer()
        result = trainer.train([])
        assert result.success is False
        assert result.error == "No training examples provided"

    def test_successful_run_writes_metadata(self, monkeypatch, tmp_path, examples):
        monkeypatch.setattr(dpo_mod, "_check_dpo_dependencies", lambda: (True, "trl + peft"))
        cfg = DPOConfig(output_dir=str(tmp_path / "dpo_out"))
        trainer = DPOTrainer(config=cfg)

        captured = {}

        def fake_run(exs, out_dir):
            captured["examples"] = exs
            captured["out_dir"] = out_dir
            return DPOTrainResult(
                success=True,
                output_dir=str(out_dir),
                adapter_path=str(out_dir / "adapter"),
                final_loss=0.25,
                rewards_chosen=1.0,
                rewards_rejected=0.4,
                reward_margin=0.6,
                total_steps=10,
            )

        monkeypatch.setattr(trainer, "_run_dpo", fake_run)
        result = trainer.train(examples)

        assert result.success is True
        assert result.total_steps == 10
        # train() sets the elapsed time
        assert result.train_time_seconds >= 0.0
        # output dir created
        assert (tmp_path / "dpo_out").is_dir()
        # _run_dpo got the right args
        assert captured["examples"] is examples
        assert captured["out_dir"] == Path(cfg.output_dir)

        # metadata file written with rounded dict
        meta_path = tmp_path / "dpo_out" / "dpo_result.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["success"] is True
        assert data["total_steps"] == 10
        assert data["reward_margin"] == 0.6

    def test_run_dpo_exception_is_caught(self, monkeypatch, tmp_path, examples):
        monkeypatch.setattr(dpo_mod, "_check_dpo_dependencies", lambda: (True, "trl + peft"))
        cfg = DPOConfig(output_dir=str(tmp_path / "dpo_err"))
        trainer = DPOTrainer(config=cfg)

        def boom(exs, out_dir):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(trainer, "_run_dpo", boom)
        result = trainer.train(examples)

        assert result.success is False
        assert result.error == "CUDA out of memory"
        assert result.output_dir == str(tmp_path / "dpo_err")
        # No metadata file written on the exception path
        assert not (tmp_path / "dpo_err" / "dpo_result.json").exists()

    def test_creates_nested_output_dir(self, monkeypatch, tmp_path, examples):
        monkeypatch.setattr(dpo_mod, "_check_dpo_dependencies", lambda: (True, "trl + peft"))
        nested = tmp_path / "a" / "b" / "c"
        cfg = DPOConfig(output_dir=str(nested))
        trainer = DPOTrainer(config=cfg)
        monkeypatch.setattr(trainer, "_run_dpo", lambda exs, out: DPOTrainResult(success=True))
        trainer.train(examples)
        assert nested.is_dir()


class TestRunDPO:
    """Exercise _run_dpo's pure logic with all heavy deps stubbed."""

    def _install_stubs(self, monkeypatch, log_history, global_step):
        import sys
        import types

        captured = {}

        # datasets.Dataset
        datasets_mod = types.ModuleType("datasets")

        class FakeDataset:
            @classmethod
            def from_list(cls, rows):
                captured["rows"] = rows
                return cls()

        datasets_mod.Dataset = FakeDataset

        # peft.LoraConfig / TaskType
        peft_mod = types.ModuleType("peft")

        def lora_config(**kwargs):
            captured["lora_kwargs"] = kwargs
            return ("lora", kwargs)

        peft_mod.LoraConfig = lora_config
        peft_mod.TaskType = types.SimpleNamespace(CAUSAL_LM="CAUSAL_LM")

        # transformers
        transformers_mod = types.ModuleType("transformers")

        class FakeTokenizer:
            pad_token = None
            eos_token = "<eos>"
            saved_to = None

            def save_pretrained(self, path):
                FakeTokenizer.saved_to = path

        class FakeModel:
            saved_to = None

            def save_pretrained(self, path):
                FakeModel.saved_to = path

        fake_model = FakeModel()
        fake_tokenizer = FakeTokenizer()

        class AutoModelForCausalLM:
            @staticmethod
            def from_pretrained(name, **kwargs):
                captured["model_kwargs"] = kwargs
                captured["model_name"] = name
                return fake_model

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(name, **kwargs):
                return fake_tokenizer

        def bnb_config(**kwargs):
            captured["bnb_kwargs"] = kwargs
            return ("bnb", kwargs)

        transformers_mod.AutoModelForCausalLM = AutoModelForCausalLM
        transformers_mod.AutoTokenizer = AutoTokenizer
        transformers_mod.BitsAndBytesConfig = bnb_config

        # torch
        torch_mod = types.ModuleType("torch")
        torch_mod.bfloat16 = "bfloat16"

        # trl
        trl_mod = types.ModuleType("trl")

        def trl_dpo_config(**kwargs):
            captured["training_args"] = kwargs
            return ("args", kwargs)

        class TRLDPOTrainer:
            def __init__(self, **kwargs):
                captured["trainer_kwargs"] = kwargs
                self.state = types.SimpleNamespace(log_history=log_history, global_step=global_step)

            def train(self):
                captured["trained"] = True

        trl_mod.DPOConfig = trl_dpo_config
        trl_mod.DPOTrainer = TRLDPOTrainer

        for name, mod in {
            "datasets": datasets_mod,
            "peft": peft_mod,
            "transformers": transformers_mod,
            "torch": torch_mod,
            "trl": trl_mod,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)

        return captured, fake_model, fake_tokenizer

    def test_run_dpo_full_path(self, monkeypatch, tmp_path, examples):
        captured, model, tokenizer = self._install_stubs(
            monkeypatch,
            log_history=[
                {"loss": 9.9},
                {
                    "loss": 0.5,
                    "rewards/chosen": 2.0,
                    "rewards/rejected": 0.5,
                },
            ],
            global_step=7,
        )
        cfg = DPOConfig(output_dir=str(tmp_path / "out"), use_4bit=True)
        trainer = DPOTrainer(config=cfg)
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = trainer._run_dpo(examples, out_dir)

        assert result.success is True
        assert result.final_loss == 0.5
        assert result.rewards_chosen == 2.0
        assert result.rewards_rejected == 0.5
        assert result.reward_margin == 1.5
        assert result.total_steps == 7
        assert result.adapter_path == str(out_dir / "adapter")

        # dataset rows mapped from examples
        assert captured["rows"][0] == {
            "prompt": "What is 2+2?",
            "chosen": "The answer is 4.",
            "rejected": "I don't know.",
        }
        # training was invoked and models saved
        assert captured["trained"] is True
        assert model.saved_to == str(out_dir / "adapter")
        assert tokenizer.saved_to == str(out_dir / "adapter")
        # pad_token set from eos_token since it was None
        assert tokenizer.pad_token == "<eos>"
        # 4-bit quantization config was built
        assert "bnb_kwargs" in captured
        assert captured["bnb_kwargs"]["load_in_4bit"] is True
        assert captured["model_kwargs"]["quantization_config"] == (
            "bnb",
            captured["bnb_kwargs"],
        )
        # training args carry config values
        ta = captured["training_args"]
        assert ta["beta"] == cfg.beta
        assert ta["num_train_epochs"] == cfg.epochs
        assert ta["max_length"] == cfg.max_length

    def test_run_dpo_without_4bit_skips_bnb(self, monkeypatch, tmp_path, examples):
        captured, model, _ = self._install_stubs(monkeypatch, log_history=[], global_step=0)
        cfg = DPOConfig(output_dir=str(tmp_path / "out2"), use_4bit=False)
        trainer = DPOTrainer(config=cfg)
        out_dir = tmp_path / "out2"
        out_dir.mkdir(parents=True, exist_ok=True)

        result = trainer._run_dpo(examples, out_dir)

        assert result.success is True
        # No bnb config built; quantization_config passed as None
        assert "bnb_kwargs" not in captured
        assert captured["model_kwargs"]["quantization_config"] is None
        # Empty log_history -> metrics stay at defaults
        assert result.final_loss == 0.0
        assert result.rewards_chosen == 0.0
        assert result.rewards_rejected == 0.0
        assert result.reward_margin == 0.0
        assert result.total_steps == 0

    def test_run_dpo_existing_pad_token_preserved(self, monkeypatch, tmp_path, examples):
        import sys

        captured, _, tokenizer = self._install_stubs(monkeypatch, log_history=[], global_step=1)
        # Give the tokenizer an existing pad token before running.
        sys.modules["transformers"].AutoTokenizer.from_pretrained("x").pad_token = "<pad>"

        cfg = DPOConfig(output_dir=str(tmp_path / "out3"))
        trainer = DPOTrainer(config=cfg)
        out_dir = tmp_path / "out3"
        out_dir.mkdir(parents=True, exist_ok=True)

        trainer._run_dpo(examples, out_dir)
        # Existing pad token must not be overwritten by eos_token.
        assert tokenizer.pad_token == "<pad>"


class TestPrepareDPODataset:
    def test_writes_jsonl(self, tmp_path, examples):
        out = tmp_path / "data" / "dpo.jsonl"
        returned = prepare_dpo_dataset(examples, out)

        assert returned == out
        assert out.exists()
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first == {
            "prompt": "What is 2+2?",
            "chosen": "The answer is 4.",
            "rejected": "I don't know.",
        }
        # metadata is intentionally not serialized here
        assert "metadata" not in first

    def test_creates_parent_dirs(self, tmp_path, examples):
        out = tmp_path / "deep" / "nested" / "path" / "dpo.jsonl"
        prepare_dpo_dataset(examples, out)
        assert out.exists()

    def test_empty_examples_creates_empty_file(self, tmp_path):
        out = tmp_path / "empty.jsonl"
        prepare_dpo_dataset([], out)
        assert out.exists()
        assert out.read_text() == ""
