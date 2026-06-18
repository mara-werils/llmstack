"""Tests for llmstack.finetune.trainer."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

from llmstack.finetune.data import ChatExample
from llmstack.finetune.trainer import (
    Trainer,
    TrainConfig,
    TrainResult,
    _check_dependencies,
)


class FakeTrainerCallback:
    """Stand-in for transformers.TrainerCallback (must be a real class to subclass)."""


class FakeTrainerState:
    def __init__(self, global_step=10, max_steps=100):
        self.global_step = global_step
        self.max_steps = max_steps


class FakeSFTTrainer:
    """Captures construction kwargs and fires callbacks on .train() like real HF Trainer."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.state = FakeTrainerState()

    def train(self, resume_from_checkpoint=None):
        for cb in self.kwargs.get("callbacks", []):
            cb.on_log(None, self.state, None, logs={"loss": 0.4, "epoch": 1.0})


class FakeDataset(list):
    @classmethod
    def from_list(cls, records):
        return cls(records)

    def map(self, fn):
        return FakeDataset([{**rec, **fn(rec)} for rec in self])


class PlainTokenizer:
    """A tokenizer without apply_chat_template, to hit the manual-join fallback."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"

    def save_pretrained(self, path):
        pass


def _train_data():
    return [
        ChatExample(messages=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]),
        ChatExample(messages=[{"role": "user", "content": "bye"}]),
    ]


def test_train_result_to_dict_rounds_and_summarizes_history():
    result = TrainResult(
        success=True,
        output_dir="/out",
        adapter_path="/out/adapter",
        final_loss=0.12345,
        best_loss=0.09876,
        total_steps=42,
        train_time_seconds=12.345,
        loss_history=[{"step": 1, "loss": 0.5}],
    )
    d = result.to_dict()
    assert d["final_loss"] == 0.1235
    assert d["best_loss"] == 0.0988
    assert d["train_time_seconds"] == 12.3
    assert d["loss_history_length"] == 1


def test_check_dependencies_none_available():
    with patch.dict(sys.modules, {"unsloth": None, "peft": None, "trl": None, "transformers": None}):
        has_unsloth, has_peft, message = _check_dependencies()
    assert (has_unsloth, has_peft, message) == (False, False, "none")


def test_check_dependencies_peft_only():
    fake = MagicMock()
    with patch.dict(
        sys.modules,
        {"unsloth": None, "peft": fake, "trl": fake, "transformers": fake},
    ):
        has_unsloth, has_peft, message = _check_dependencies()
    assert (has_unsloth, has_peft, message) == (False, True, "peft + trl")


def test_check_dependencies_unsloth_available():
    fake = MagicMock()
    with patch.dict(
        sys.modules,
        {"unsloth": fake, "peft": None, "trl": None, "transformers": None},
    ):
        has_unsloth, has_peft, message = _check_dependencies()
    assert (has_unsloth, has_peft, message) == (True, False, "unsloth (fast)")


def test_set_progress_callback():
    trainer = Trainer(TrainConfig())
    cb = lambda step, total, loss: None  # noqa: E731
    trainer.set_progress_callback(cb)
    assert trainer._progress_callback is cb


def test_train_no_backends_available(tmp_path):
    trainer = Trainer(TrainConfig(output_dir=str(tmp_path / "out")))
    with patch(
        "llmstack.finetune.trainer._check_dependencies", return_value=(False, False, "none")
    ):
        result = trainer.train(_train_data())
    assert result.success is False
    assert "No training backend found" in result.error


def test_train_unsloth_backend_writes_metadata(tmp_path):
    out_dir = tmp_path / "out"
    trainer = Trainer(TrainConfig(output_dir=str(out_dir)))
    fake_result = TrainResult(success=True, output_dir=str(out_dir), adapter_path=str(out_dir / "adapter"))

    with (
        patch(
            "llmstack.finetune.trainer._check_dependencies",
            return_value=(True, False, "unsloth (fast)"),
        ),
        patch.object(trainer, "_train_unsloth", return_value=fake_result) as mock_train,
    ):
        result = trainer.train(_train_data())

    mock_train.assert_called_once()
    assert result.success is True
    assert result.train_time_seconds >= 0
    meta = json.loads((out_dir / "train_result.json").read_text())
    assert meta["success"] is True


def test_train_peft_backend_used_when_no_unsloth(tmp_path):
    out_dir = tmp_path / "out"
    trainer = Trainer(TrainConfig(output_dir=str(out_dir)))
    fake_result = TrainResult(success=True, output_dir=str(out_dir))

    with (
        patch(
            "llmstack.finetune.trainer._check_dependencies",
            return_value=(False, True, "peft + trl"),
        ),
        patch.object(trainer, "_train_peft", return_value=fake_result) as mock_train,
    ):
        result = trainer.train(_train_data())

    mock_train.assert_called_once()
    assert result.success is True


def test_train_handles_exception_from_backend(tmp_path):
    out_dir = tmp_path / "out"
    trainer = Trainer(TrainConfig(output_dir=str(out_dir)))

    with (
        patch(
            "llmstack.finetune.trainer._check_dependencies",
            return_value=(True, False, "unsloth (fast)"),
        ),
        patch.object(trainer, "_train_unsloth", side_effect=RuntimeError("OOM")),
    ):
        result = trainer.train(_train_data())

    assert result.success is False
    assert result.error == "OOM"
    assert result.output_dir == str(out_dir)


def test_train_unsloth_full_flow(tmp_path):
    fake_model = MagicMock()
    fake_tokenizer = MagicMock()  # has apply_chat_template by default -> chat-template branch

    fake_fast_lm = MagicMock()
    fake_fast_lm.from_pretrained.return_value = (fake_model, fake_tokenizer)
    fake_fast_lm.get_peft_model.return_value = fake_model
    fake_unsloth = MagicMock(FastLanguageModel=fake_fast_lm)

    fake_trl = MagicMock(SFTTrainer=FakeSFTTrainer)
    fake_transformers = MagicMock(
        TrainingArguments=MagicMock(return_value=MagicMock()),
        TrainerCallback=FakeTrainerCallback,
    )
    fake_datasets = MagicMock(Dataset=FakeDataset)

    trainer = Trainer(TrainConfig(output_dir=str(tmp_path)))
    trainer.set_progress_callback(lambda step, total, loss: None)

    with patch.dict(
        sys.modules,
        {
            "unsloth": fake_unsloth,
            "trl": fake_trl,
            "transformers": fake_transformers,
            "datasets": fake_datasets,
        },
    ):
        result = trainer._train_unsloth(_train_data(), _train_data(), tmp_path)

    assert result.success is True
    assert result.total_steps == 10
    assert result.final_loss == 0.4
    assert result.best_loss == 0.4
    assert trainer._loss_history == [{"step": 10, "loss": 0.4, "epoch": 1.0}]
    fake_model.save_pretrained.assert_called_once()
    fake_tokenizer.save_pretrained.assert_called_once()


def test_train_peft_full_flow_manual_join_and_quantization(tmp_path):
    fake_base_model = MagicMock()
    fake_auto_model = MagicMock(from_pretrained=MagicMock(return_value=fake_base_model))
    fake_tokenizer = PlainTokenizer()
    fake_auto_tokenizer = MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer))

    fake_peft_model = MagicMock()
    fake_peft = MagicMock(
        LoraConfig=MagicMock(return_value=MagicMock()),
        get_peft_model=MagicMock(return_value=fake_peft_model),
        TaskType=MagicMock(CAUSAL_LM="CAUSAL_LM"),
    )

    fake_trl = MagicMock(SFTTrainer=FakeSFTTrainer)
    fake_transformers = MagicMock(
        AutoModelForCausalLM=fake_auto_model,
        AutoTokenizer=fake_auto_tokenizer,
        TrainingArguments=MagicMock(return_value=MagicMock()),
        BitsAndBytesConfig=MagicMock(return_value=MagicMock()),
        TrainerCallback=FakeTrainerCallback,
    )
    fake_datasets = MagicMock(Dataset=FakeDataset)
    fake_torch = MagicMock()

    trainer = Trainer(TrainConfig(output_dir=str(tmp_path)))
    trainer.set_progress_callback(lambda step, total, loss: None)

    with patch.dict(
        sys.modules,
        {
            "peft": fake_peft,
            "trl": fake_trl,
            "transformers": fake_transformers,
            "datasets": fake_datasets,
            "torch": fake_torch,
        },
    ):
        result = trainer._train_peft(_train_data(), None, tmp_path)

    assert result.success is True
    assert result.total_steps == 10
    assert result.final_loss == 0.4
    assert result.best_loss == 0.4
    assert fake_tokenizer.pad_token == "<eos>"  # filled in from eos_token
    fake_peft_model.save_pretrained.assert_called_once()
