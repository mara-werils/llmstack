"""Training engine — LoRA/QLoRA fine-tuning with progress tracking.

Wraps unsloth (preferred) or HuggingFace PEFT/TRL for the actual training.
Falls back gracefully when optional dependencies are missing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from llmstack.finetune.data import ChatExample
from llmstack.finetune.hyperparams import TrainHyperparams

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Full training configuration."""

    base_model: str = "unsloth/llama-3.2-1b-instruct-bnb-4bit"
    method: str = "qlora"               # "qlora", "lora", "full"
    output_dir: str = "./finetune-output"
    hyperparams: TrainHyperparams = field(default_factory=TrainHyperparams)
    resume_from: str | None = None


@dataclass
class TrainResult:
    """Result of a training run."""

    success: bool = False
    output_dir: str = ""
    adapter_path: str = ""
    final_loss: float = 0.0
    best_loss: float = 0.0
    total_steps: int = 0
    train_time_seconds: float = 0.0
    loss_history: list[dict[str, float]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "adapter_path": self.adapter_path,
            "final_loss": round(self.final_loss, 4),
            "best_loss": round(self.best_loss, 4),
            "total_steps": self.total_steps,
            "train_time_seconds": round(self.train_time_seconds, 1),
            "loss_history_length": len(self.loss_history),
            "error": self.error,
        }


def _check_dependencies() -> tuple[bool, bool, str]:
    """Check which training backends are available.

    Returns (has_unsloth, has_peft, message).
    """
    has_unsloth = False
    has_peft = False

    try:
        import unsloth  # noqa: F401
        has_unsloth = True
    except ImportError:
        pass

    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
        import transformers  # noqa: F401
        has_peft = True
    except ImportError:
        pass

    if has_unsloth:
        return True, has_peft, "unsloth (fast)"
    elif has_peft:
        return False, True, "peft + trl"
    else:
        return False, False, "none"


class Trainer:
    """Fine-tuning trainer with progress callbacks.

    Supports two backends:
    - unsloth: 2x faster training with optimized kernels
    - peft + trl: standard HuggingFace fine-tuning

    Falls back gracefully when dependencies are missing, providing
    clear installation instructions.
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self._loss_history: list[dict[str, float]] = []
        self._progress_callback: Callable | None = None

    def set_progress_callback(self, callback: Callable[[int, int, float], None]) -> None:
        """Set a callback for progress updates: callback(step, total_steps, loss)."""
        self._progress_callback = callback

    def train(
        self,
        train_data: list[ChatExample],
        eval_data: list[ChatExample] | None = None,
    ) -> TrainResult:
        """Run fine-tuning.

        Returns TrainResult with loss history and output paths.
        """
        has_unsloth, has_peft, backend = _check_dependencies()

        if not has_unsloth and not has_peft:
            return TrainResult(
                success=False,
                error=(
                    "No training backend found. Install one of:\n"
                    "  pip install 'unsloth[cu121]'    # recommended, 2x faster\n"
                    "  pip install peft trl transformers datasets  # standard"
                ),
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()

        try:
            if has_unsloth:
                result = self._train_unsloth(train_data, eval_data, output_dir)
            else:
                result = self._train_peft(train_data, eval_data, output_dir)
        except Exception as exc:
            logger.error("Training failed: %s", exc)
            return TrainResult(
                success=False,
                output_dir=str(output_dir),
                error=str(exc),
                loss_history=self._loss_history,
            )

        result.train_time_seconds = time.monotonic() - t0
        result.loss_history = self._loss_history

        # Save training metadata
        meta_path = output_dir / "train_result.json"
        meta_path.write_text(json.dumps(result.to_dict(), indent=2))

        return result

    def _train_unsloth(
        self, train_data: list[ChatExample],
        eval_data: list[ChatExample] | None,
        output_dir: Path,
    ) -> TrainResult:
        """Train using the unsloth library (2x faster)."""
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments

        hp = self.config.hyperparams

        # Load model with unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.base_model,
            max_seq_length=hp.max_seq_length,
            load_in_4bit=hp.use_4bit,
            dtype=None,
        )

        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=hp.target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            use_gradient_checkpointing="unsloth",
        )

        # Prepare dataset
        train_dataset = self._to_hf_dataset(train_data, tokenizer)
        eval_dataset = self._to_hf_dataset(eval_data, tokenizer) if eval_data else None

        # Custom callback for progress
        callbacks = []
        if self._progress_callback:
            from transformers import TrainerCallback

            outer = self

            class ProgressCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and "loss" in logs:
                        outer._loss_history.append({
                            "step": state.global_step,
                            "loss": logs["loss"],
                            "epoch": logs.get("epoch", 0),
                        })
                        if outer._progress_callback:
                            outer._progress_callback(
                                state.global_step,
                                state.max_steps,
                                logs["loss"],
                            )

            callbacks.append(ProgressCallback())

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=hp.epochs,
            per_device_train_batch_size=hp.batch_size,
            gradient_accumulation_steps=hp.gradient_accumulation_steps,
            learning_rate=hp.learning_rate,
            warmup_ratio=hp.warmup_ratio,
            weight_decay=hp.weight_decay,
            lr_scheduler_type=hp.lr_scheduler,
            logging_steps=hp.logging_steps,
            save_steps=hp.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=hp.eval_steps if eval_dataset else None,
            save_total_limit=3,
            fp16=False,
            bf16=True,
            report_to="none",
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            callbacks=callbacks,
            max_seq_length=hp.max_seq_length,
        )

        trainer.train(resume_from_checkpoint=self.config.resume_from)

        # Save adapter
        adapter_path = output_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        # Get final loss
        final_loss = 0.0
        best_loss = float("inf")
        if self._loss_history:
            final_loss = self._loss_history[-1]["loss"]
            best_loss = min(h["loss"] for h in self._loss_history)

        return TrainResult(
            success=True,
            output_dir=str(output_dir),
            adapter_path=str(adapter_path),
            final_loss=final_loss,
            best_loss=best_loss,
            total_steps=trainer.state.global_step,
        )

    def _train_peft(
        self, train_data: list[ChatExample],
        eval_data: list[ChatExample] | None,
        output_dir: Path,
    ) -> TrainResult:
        """Train using standard PEFT + TRL (fallback)."""
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
            BitsAndBytesConfig,
        )

        hp = self.config.hyperparams

        # Quantization config
        bnb_config = None
        if hp.use_4bit:
            import torch
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, hp.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=hp.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # LoRA config
        lora_config = LoraConfig(
            r=hp.lora_r,
            lora_alpha=hp.lora_alpha,
            lora_dropout=hp.lora_dropout,
            target_modules=hp.target_modules or [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

        # Prepare datasets
        train_dataset = self._to_hf_dataset(train_data, tokenizer)
        eval_dataset = self._to_hf_dataset(eval_data, tokenizer) if eval_data else None

        callbacks = []
        if self._progress_callback:
            from transformers import TrainerCallback

            outer = self

            class ProgressCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs and "loss" in logs:
                        outer._loss_history.append({
                            "step": state.global_step,
                            "loss": logs["loss"],
                        })
                        if outer._progress_callback:
                            outer._progress_callback(
                                state.global_step,
                                state.max_steps,
                                logs["loss"],
                            )

            callbacks.append(ProgressCallback())

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=hp.epochs,
            per_device_train_batch_size=hp.batch_size,
            gradient_accumulation_steps=hp.gradient_accumulation_steps,
            learning_rate=hp.learning_rate,
            warmup_ratio=hp.warmup_ratio,
            weight_decay=hp.weight_decay,
            lr_scheduler_type=hp.lr_scheduler,
            logging_steps=hp.logging_steps,
            save_steps=hp.save_steps,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=hp.eval_steps if eval_dataset else None,
            save_total_limit=3,
            report_to="none",
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            peft_config=lora_config,
            callbacks=callbacks,
            max_seq_length=hp.max_seq_length,
        )

        trainer.train(resume_from_checkpoint=self.config.resume_from)

        # Save
        adapter_path = output_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        final_loss = 0.0
        best_loss = float("inf")
        if self._loss_history:
            final_loss = self._loss_history[-1]["loss"]
            best_loss = min(h["loss"] for h in self._loss_history)

        return TrainResult(
            success=True,
            output_dir=str(output_dir),
            adapter_path=str(adapter_path),
            final_loss=final_loss,
            best_loss=best_loss,
            total_steps=trainer.state.global_step,
        )

    def _to_hf_dataset(self, examples: list[ChatExample], tokenizer: Any) -> Any:
        """Convert ChatExamples to a HuggingFace Dataset."""
        from datasets import Dataset

        def _format_chat(example: dict) -> dict:
            messages = example["messages"]
            if hasattr(tokenizer, "apply_chat_template"):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
            else:
                text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return {"text": text}

        records = [ex.to_dict() for ex in examples]
        dataset = Dataset.from_list(records)
        return dataset.map(_format_chat)
