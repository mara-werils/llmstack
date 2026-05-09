"""Export — convert fine-tuned adapter to GGUF and create Ollama model.

Two-step process:
1. Merge adapter into base model and convert to GGUF format
2. Create an Ollama model from the GGUF file via Modelfile
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    """Result of a GGUF export."""

    success: bool = False
    gguf_path: str = ""
    ollama_model: str = ""
    size_mb: float = 0.0
    quantization: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "gguf_path": self.gguf_path,
            "ollama_model": self.ollama_model,
            "size_mb": round(self.size_mb, 1),
            "quantization": self.quantization,
            "error": self.error,
        }


def export_gguf(
    adapter_path: str,
    base_model: str,
    output_path: str | None = None,
    quantization: str = "q4_k_m",
) -> ExportResult:
    """Export a fine-tuned adapter to GGUF format.

    Attempts multiple methods in order:
    1. unsloth's built-in GGUF export (fastest)
    2. llama.cpp convert script
    3. Manual merge + convert
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.is_dir():
        return ExportResult(success=False, error=f"Adapter path not found: {adapter_path}")

    output = Path(output_path) if output_path else adapter_dir.parent / "model.gguf"

    # Try unsloth export first
    result = _export_via_unsloth(adapter_dir, base_model, output, quantization)
    if result.success:
        return result

    # Try llama.cpp
    result = _export_via_llamacpp(adapter_dir, base_model, output, quantization)
    if result.success:
        return result

    return ExportResult(
        success=False,
        error=(
            "Could not export to GGUF. Install one of:\n"
            "  pip install 'unsloth[cu121]'  # includes GGUF export\n"
            "  brew install llama.cpp        # or build from source"
        ),
    )


def _export_via_unsloth(
    adapter_dir: Path, base_model: str, output: Path, quantization: str,
) -> ExportResult:
    """Export using unsloth's save_pretrained_gguf."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=2048,
            load_in_4bit=True,
        )

        model.save_pretrained_gguf(
            str(output.parent),
            tokenizer,
            quantization_method=quantization,
        )

        gguf_files = list(output.parent.glob("*.gguf"))
        if gguf_files:
            actual_path = gguf_files[0]
            size_mb = actual_path.stat().st_size / (1024 * 1024)
            return ExportResult(
                success=True,
                gguf_path=str(actual_path),
                quantization=quantization,
                size_mb=size_mb,
            )

        return ExportResult(success=False, error="GGUF file not created by unsloth")

    except ImportError:
        return ExportResult(success=False, error="unsloth not installed")
    except Exception as exc:
        return ExportResult(success=False, error=f"unsloth export failed: {exc}")


def _export_via_llamacpp(
    adapter_dir: Path, base_model: str, output: Path, quantization: str,
) -> ExportResult:
    """Export using llama.cpp's convert tool."""
    # Check if llama-quantize is available
    try:
        subprocess.run(["llama-quantize", "--help"], capture_output=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ExportResult(success=False, error="llama.cpp not found")

    try:
        # First merge adapter with base model
        merged_dir = adapter_dir.parent / "merged"
        _merge_adapter(adapter_dir, base_model, merged_dir)

        # Convert to GGUF
        convert_cmd = [
            "python3", "-m", "llama_cpp.convert",
            str(merged_dir), "--outfile", str(output), "--outtype", "f16",
        ]
        subprocess.run(convert_cmd, capture_output=True, text=True, timeout=600, check=True)

        # Quantize
        if quantization != "f16":
            quantized = output.with_suffix(f".{quantization}.gguf")
            quant_cmd = ["llama-quantize", str(output), str(quantized), quantization]
            subprocess.run(quant_cmd, capture_output=True, text=True, timeout=600, check=True)
            output.unlink(missing_ok=True)
            output = quantized

        size_mb = output.stat().st_size / (1024 * 1024)
        return ExportResult(
            success=True, gguf_path=str(output),
            quantization=quantization, size_mb=size_mb,
        )

    except Exception as exc:
        return ExportResult(success=False, error=f"llama.cpp export failed: {exc}")


def _merge_adapter(adapter_dir: Path, base_model: str, output_dir: Path) -> None:
    """Merge LoRA adapter back into the base model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="cpu")
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(output_dir))


def create_ollama_model(
    gguf_path: str,
    model_name: str,
    system_prompt: str = "",
    ollama_url: str = "http://localhost:11434",
) -> ExportResult:
    """Create an Ollama model from a GGUF file using a Modelfile.

    Generates a Modelfile, runs `ollama create`, and the model becomes
    immediately available via `ollama run <model_name>`.
    """
    gguf = Path(gguf_path)
    if not gguf.is_file():
        return ExportResult(success=False, error=f"GGUF file not found: {gguf_path}")

    # Write Modelfile
    modelfile_content = f'FROM {gguf.resolve()}\n'
    if system_prompt:
        escaped = system_prompt.replace('"', '\\"')
        modelfile_content += f'SYSTEM "{escaped}"\n'

    # Standard parameters
    modelfile_content += 'PARAMETER temperature 0.7\n'
    modelfile_content += 'PARAMETER top_p 0.9\n'
    modelfile_content += 'PARAMETER stop "<|eot_id|>"\n'
    modelfile_content += 'PARAMETER stop "<|end_of_text|>"\n'

    modelfile_path = gguf.parent / "Modelfile"
    modelfile_path.write_text(modelfile_content)

    # Run ollama create
    try:
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True, text=True, timeout=300,
        )

        if result.returncode != 0:
            return ExportResult(
                success=False,
                error=f"ollama create failed: {result.stderr.strip()}",
                gguf_path=str(gguf),
            )

        size_mb = gguf.stat().st_size / (1024 * 1024)
        return ExportResult(
            success=True,
            gguf_path=str(gguf),
            ollama_model=model_name,
            size_mb=size_mb,
        )

    except FileNotFoundError:
        return ExportResult(
            success=False,
            error="ollama not found. Install: https://ollama.ai",
            gguf_path=str(gguf),
        )
    except subprocess.TimeoutExpired:
        return ExportResult(
            success=False, error="ollama create timed out",
            gguf_path=str(gguf),
        )
