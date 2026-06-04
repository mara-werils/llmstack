"""Auto hyperparameter selection based on dataset size and model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainHyperparams:
    """Training hyperparameters for LoRA/QLoRA fine-tuning."""

    # LoRA params
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha (usually 2x rank)
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None  # auto-detect if None

    # Training params
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    lr_scheduler: str = "cosine"

    # QLoRA specific
    use_4bit: bool = True  # QLoRA 4-bit quantization
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # Saving
    save_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 50

    def to_dict(self) -> dict:
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_seq_length": self.max_seq_length,
            "lr_scheduler": self.lr_scheduler,
            "use_4bit": self.use_4bit,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
        }


def auto_hyperparams(
    num_examples: int,
    model_size_b: float = 7.0,
    method: str = "qlora",
    max_seq_length: int = 2048,
) -> TrainHyperparams:
    """Auto-select training hyperparameters based on dataset size and model.

    Heuristics:
    - Small datasets (<500): fewer epochs, lower LR, smaller rank
    - Medium datasets (500-5000): standard settings
    - Large datasets (>5000): more aggressive, higher rank
    - Larger models: smaller batch size, lower LR
    """
    params = TrainHyperparams(max_seq_length=max_seq_length)

    # Method
    params.use_4bit = method in ("qlora", "4bit")

    # --- Epochs based on dataset size ---
    if num_examples < 100:
        params.epochs = 5
    elif num_examples < 500:
        params.epochs = 3
    elif num_examples < 5000:
        params.epochs = 2
    else:
        params.epochs = 1

    # --- LoRA rank based on model size and dataset ---
    if model_size_b >= 30:
        params.lora_r = 32
        params.lora_alpha = 64
    elif model_size_b >= 7:
        params.lora_r = 16
        params.lora_alpha = 32
    else:
        params.lora_r = 8
        params.lora_alpha = 16

    if num_examples > 10000:
        params.lora_r = min(64, params.lora_r * 2)
        params.lora_alpha = params.lora_r * 2

    # --- Batch size based on model size ---
    if model_size_b >= 30:
        params.batch_size = 1
        params.gradient_accumulation_steps = 16
    elif model_size_b >= 13:
        params.batch_size = 2
        params.gradient_accumulation_steps = 8
    elif model_size_b >= 7:
        params.batch_size = 4
        params.gradient_accumulation_steps = 4
    else:
        params.batch_size = 8
        params.gradient_accumulation_steps = 2

    # --- Learning rate based on model size ---
    if model_size_b >= 30:
        params.learning_rate = 1e-4
    elif model_size_b >= 13:
        params.learning_rate = 1.5e-4
    else:
        params.learning_rate = 2e-4

    # Small datasets need lower LR
    if num_examples < 100:
        params.learning_rate *= 0.5

    # --- Logging/save frequency ---
    total_steps = (num_examples * params.epochs) // (
        params.batch_size * params.gradient_accumulation_steps
    )
    total_steps = max(total_steps, 1)

    params.logging_steps = max(1, total_steps // 50)
    params.save_steps = max(10, total_steps // 5)
    params.eval_steps = max(10, total_steps // 10)

    return params


def estimate_model_size(model_name: str) -> float:
    """Estimate model size in billions from the model name."""
    name = model_name.lower()

    size_patterns = [
        ("405b", 405.0),
        ("70b", 70.0),
        ("34b", 34.0),
        ("33b", 33.0),
        ("30b", 30.0),
        ("27b", 27.0),
        ("14b", 14.0),
        ("13b", 13.0),
        ("8b", 8.0),
        ("7b", 7.0),
        ("3b", 3.0),
        ("2b", 2.0),
        ("1b", 1.0),
        ("0.5b", 0.5),
        (":1b", 1.0),
        (":3b", 3.0),
        (":8b", 8.0),
        (":70b", 70.0),
    ]

    for pattern, size in size_patterns:
        if pattern in name:
            return size

    # Default guess
    return 7.0
