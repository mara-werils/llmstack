"""DPO trainer — Direct Preference Optimization for learning from preferences.

When users provide corrections or A/B preferences, DPO training aligns
the model with those preferences without needing a reward model.
This is more sample-efficient than SFT for preference data.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from llmstack.learn.dataset import DPOExample

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Configuration for DPO training."""

    base_model: str = "unsloth/llama-3.2-1b-instruct-bnb-4bit"
    output_dir: str = str(Path.home() / ".llmstack" / "training" / "dpo")
    beta: float = 0.1  # DPO temperature parameter
    learning_rate: float = 5e-6
    epochs: int = 1
    batch_size: int = 2
    max_length: int = 1024
    max_prompt_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    lora_r: int = 16
    lora_alpha: int = 32
    use_4bit: bool = True


@dataclass
class DPOTrainResult:
    """Result of a DPO training run."""

    success: bool = False
    output_dir: str = ""
    adapter_path: str = ""
    final_loss: float = 0.0
    rewards_chosen: float = 0.0
    rewards_rejected: float = 0.0
    reward_margin: float = 0.0
    train_time_seconds: float = 0.0
    total_steps: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "adapter_path": self.adapter_path,
            "final_loss": round(self.final_loss, 4),
            "rewards_chosen": round(self.rewards_chosen, 4),
            "rewards_rejected": round(self.rewards_rejected, 4),
            "reward_margin": round(self.reward_margin, 4),
            "train_time_seconds": round(self.train_time_seconds, 1),
            "total_steps": self.total_steps,
            "error": self.error,
        }


def _check_dpo_dependencies() -> tuple[bool, str]:
    """Check if DPO training dependencies are available."""
    try:
        import trl  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401

        return True, "trl + peft"
    except ImportError:
        return False, "missing"


class DPOTrainer:
    """DPO (Direct Preference Optimization) trainer.

    Trains a model to prefer chosen responses over rejected ones,
    learning directly from user preference pairs without a separate
    reward model.
    """

    def __init__(self, config: DPOConfig | None = None):
        self.config = config or DPOConfig()
        self._progress_callback: Callable[[int, int, float], None] | None = None

    def set_progress_callback(self, callback: Callable[[int, int, float], None]) -> None:
        """Set progress callback: callback(step, total_steps, loss)."""
        self._progress_callback = callback

    def train(self, examples: list[DPOExample]) -> DPOTrainResult:
        """Run DPO training on preference pairs.

        Args:
            examples: List of (prompt, chosen, rejected) preference pairs
        """
        has_deps, backend = _check_dpo_dependencies()
        if not has_deps:
            return DPOTrainResult(
                success=False,
                error=(
                    "DPO training requires: pip install trl peft transformers\n"
                    "For faster training: pip install 'unsloth[cu121]'"
                ),
            )

        if not examples:
            return DPOTrainResult(success=False, error="No training examples provided")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()

        try:
            result = self._run_dpo(examples, output_dir)
        except Exception as exc:
            logger.error("DPO training failed: %s", exc)
            return DPOTrainResult(
                success=False,
                output_dir=str(output_dir),
                error=str(exc),
            )

        result.train_time_seconds = time.monotonic() - t0

        # Save metadata
        meta_path = output_dir / "dpo_result.json"
        meta_path.write_text(json.dumps(result.to_dict(), indent=2))

        return result

    def _run_dpo(self, examples: list[DPOExample], output_dir: Path) -> DPOTrainResult:
        """Execute DPO training."""
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOTrainer as TRLDPOTrainer, DPOConfig as TRLDPOConfig

        cfg = self.config

        # Quantization
        bnb_config = None
        if cfg.use_4bit:
            import torch

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # LoRA config
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

        # Prepare dataset
        dataset = Dataset.from_list(
            [
                {
                    "prompt": ex.prompt,
                    "chosen": ex.chosen,
                    "rejected": ex.rejected,
                }
                for ex in examples
            ]
        )

        # DPO training config
        training_args = TRLDPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            beta=cfg.beta,
            max_length=cfg.max_length,
            max_prompt_length=cfg.max_prompt_length,
            logging_steps=1,
            save_steps=50,
            bf16=True,
            report_to="none",
            remove_unused_columns=False,
        )

        # Train
        trainer = TRLDPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

        trainer.train()

        # Save adapter
        adapter_path = output_dir / "adapter"
        model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        # Extract metrics
        final_loss = 0.0
        rewards_chosen = 0.0
        rewards_rejected = 0.0

        if trainer.state.log_history:
            last_log = trainer.state.log_history[-1]
            final_loss = last_log.get("loss", 0.0)
            rewards_chosen = last_log.get("rewards/chosen", 0.0)
            rewards_rejected = last_log.get("rewards/rejected", 0.0)

        return DPOTrainResult(
            success=True,
            output_dir=str(output_dir),
            adapter_path=str(adapter_path),
            final_loss=final_loss,
            rewards_chosen=rewards_chosen,
            rewards_rejected=rewards_rejected,
            reward_margin=rewards_chosen - rewards_rejected,
            total_steps=trainer.state.global_step,
        )


def prepare_dpo_dataset(examples: list[DPOExample], output_path: Path) -> Path:
    """Save DPO examples in a standard format for external training."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(
                json.dumps(
                    {
                        "prompt": ex.prompt,
                        "chosen": ex.chosen,
                        "rejected": ex.rejected,
                    }
                )
                + "\n"
            )
    return output_path
