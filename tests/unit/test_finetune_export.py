"""Tests for llmstack.finetune.export."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmstack.finetune.export import (
    ExportResult,
    _export_via_llamacpp,
    _export_via_unsloth,
    _merge_adapter,
    create_ollama_model,
    export_gguf,
)


def test_export_result_to_dict_rounds_size():
    result = ExportResult(
        success=True, gguf_path="/a/b.gguf", ollama_model="m", size_mb=12.3456, quantization="q4_k_m"
    )
    assert result.to_dict() == {
        "success": True,
        "gguf_path": "/a/b.gguf",
        "ollama_model": "m",
        "size_mb": 12.3,
        "quantization": "q4_k_m",
        "error": None,
    }


def test_export_gguf_missing_adapter_dir(tmp_path):
    result = export_gguf(str(tmp_path / "missing"), "base-model")
    assert result.success is False
    assert "Adapter path not found" in result.error


def test_export_gguf_uses_unsloth_when_successful(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with patch(
        "llmstack.finetune.export._export_via_unsloth",
        return_value=ExportResult(success=True, gguf_path="x.gguf"),
    ) as mock_unsloth, patch("llmstack.finetune.export._export_via_llamacpp") as mock_llamacpp:
        result = export_gguf(str(adapter_dir), "base-model")

    assert result.success is True
    mock_unsloth.assert_called_once()
    mock_llamacpp.assert_not_called()


def test_export_gguf_falls_back_to_llamacpp(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with (
        patch(
            "llmstack.finetune.export._export_via_unsloth",
            return_value=ExportResult(success=False, error="no unsloth"),
        ),
        patch(
            "llmstack.finetune.export._export_via_llamacpp",
            return_value=ExportResult(success=True, gguf_path="y.gguf"),
        ) as mock_llamacpp,
    ):
        result = export_gguf(str(adapter_dir), "base-model", output_path=str(tmp_path / "out.gguf"))

    assert result.success is True
    mock_llamacpp.assert_called_once()


def test_export_gguf_both_methods_fail(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    with (
        patch(
            "llmstack.finetune.export._export_via_unsloth",
            return_value=ExportResult(success=False, error="no unsloth"),
        ),
        patch(
            "llmstack.finetune.export._export_via_llamacpp",
            return_value=ExportResult(success=False, error="no llama.cpp"),
        ),
    ):
        result = export_gguf(str(adapter_dir), "base-model")

    assert result.success is False
    assert "Could not export to GGUF" in result.error


def test_export_via_unsloth_not_installed(tmp_path):
    with patch.dict(sys.modules, {"unsloth": None}):
        result = _export_via_unsloth(tmp_path, "base-model", tmp_path / "out.gguf", "q4_k_m")
    assert result.success is False
    assert result.error == "unsloth not installed"


def test_export_via_unsloth_success(tmp_path):
    output = tmp_path / "out.gguf"
    output.parent.mkdir(parents=True, exist_ok=True)

    def fake_save_pretrained_gguf(out_dir, tokenizer, quantization_method):
        (Path(out_dir) / "model.gguf").write_bytes(b"x" * 2048)

    fake_model = MagicMock()
    fake_model.save_pretrained_gguf.side_effect = fake_save_pretrained_gguf
    fake_fast_lm = MagicMock()
    fake_fast_lm.from_pretrained.return_value = (fake_model, MagicMock())
    fake_unsloth = MagicMock(FastLanguageModel=fake_fast_lm)

    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        result = _export_via_unsloth(tmp_path, "base-model", output, "q4_k_m")

    assert result.success is True
    assert result.quantization == "q4_k_m"
    assert result.size_mb == 2048 / (1024 * 1024)


def test_export_via_unsloth_no_gguf_produced(tmp_path):
    fake_model = MagicMock()
    fake_fast_lm = MagicMock()
    fake_fast_lm.from_pretrained.return_value = (fake_model, MagicMock())
    fake_unsloth = MagicMock(FastLanguageModel=fake_fast_lm)

    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        result = _export_via_unsloth(tmp_path, "base-model", tmp_path / "out.gguf", "q4_k_m")

    assert result.success is False
    assert "GGUF file not created" in result.error


def test_export_via_unsloth_unexpected_exception(tmp_path):
    fake_fast_lm = MagicMock()
    fake_fast_lm.from_pretrained.side_effect = RuntimeError("boom")
    fake_unsloth = MagicMock(FastLanguageModel=fake_fast_lm)

    with patch.dict(sys.modules, {"unsloth": fake_unsloth}):
        result = _export_via_unsloth(tmp_path, "base-model", tmp_path / "out.gguf", "q4_k_m")

    assert result.success is False
    assert "unsloth export failed: boom" in result.error


def test_export_via_llamacpp_not_found(tmp_path):
    with patch("subprocess.run", side_effect=FileNotFoundError()):
        result = _export_via_llamacpp(tmp_path, "base-model", tmp_path / "out.gguf", "q4_k_m")
    assert result.success is False
    assert result.error == "llama.cpp not found"


def test_export_via_llamacpp_timeout_on_probe(tmp_path):
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=10)):
        result = _export_via_llamacpp(tmp_path, "base-model", tmp_path / "out.gguf", "q4_k_m")
    assert result.success is False
    assert result.error == "llama.cpp not found"


def test_export_via_llamacpp_success_f16(tmp_path):
    output = tmp_path / "out.gguf"

    def fake_run(cmd, **kwargs):
        if cmd[0] == "llama-quantize" and "--help" in cmd:
            return MagicMock(returncode=0)
        if "llama_cpp.convert" in cmd:
            output.write_bytes(b"x" * 4096)
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("llmstack.finetune.export._merge_adapter"),
    ):
        result = _export_via_llamacpp(tmp_path, "base-model", output, "f16")

    assert result.success is True
    assert result.quantization == "f16"
    assert result.gguf_path == str(output)


def test_export_via_llamacpp_success_with_quantization(tmp_path):
    output = tmp_path / "out.gguf"
    quantized = output.with_suffix(".q4_k_m.gguf")

    def fake_run(cmd, **kwargs):
        if cmd[0] == "llama-quantize" and "--help" in cmd:
            return MagicMock(returncode=0)
        if "llama_cpp.convert" in cmd:
            output.write_bytes(b"x" * 4096)
            return MagicMock(returncode=0)
        if cmd[0] == "llama-quantize":
            quantized.write_bytes(b"y" * 1024)
            return MagicMock(returncode=0)
        raise AssertionError(f"unexpected command: {cmd}")

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("llmstack.finetune.export._merge_adapter"),
    ):
        result = _export_via_llamacpp(tmp_path, "base-model", output, "q4_k_m")

    assert result.success is True
    assert result.gguf_path == str(quantized)
    assert not output.exists()


def test_export_via_llamacpp_convert_failure(tmp_path):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "llama-quantize" and "--help" in cmd:
            return MagicMock(returncode=0)
        raise subprocess.CalledProcessError(1, cmd)

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("llmstack.finetune.export._merge_adapter"),
    ):
        result = _export_via_llamacpp(tmp_path, "base-model", tmp_path / "out.gguf", "q4_k_m")

    assert result.success is False
    assert "llama.cpp export failed" in result.error


def test_merge_adapter_calls_save_pretrained(tmp_path):
    fake_base = MagicMock()
    fake_auto_model = MagicMock()
    fake_auto_model.from_pretrained.return_value = fake_base

    fake_merged = MagicMock()
    fake_peft_model = MagicMock()
    fake_peft_model.from_pretrained.return_value = MagicMock(merge_and_unload=MagicMock(return_value=fake_merged))

    fake_tokenizer = MagicMock()
    fake_auto_tokenizer = MagicMock()
    fake_auto_tokenizer.from_pretrained.return_value = fake_tokenizer

    fake_peft = MagicMock(PeftModel=fake_peft_model)
    fake_transformers = MagicMock(
        AutoModelForCausalLM=fake_auto_model, AutoTokenizer=fake_auto_tokenizer
    )

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    output_dir = tmp_path / "merged"

    with patch.dict(sys.modules, {"peft": fake_peft, "transformers": fake_transformers}):
        _merge_adapter(adapter_dir, "base-model", output_dir)

    assert output_dir.is_dir()
    fake_merged.save_pretrained.assert_called_once_with(str(output_dir))
    fake_tokenizer.save_pretrained.assert_called_once_with(str(output_dir))


def test_create_ollama_model_gguf_not_found(tmp_path):
    result = create_ollama_model(str(tmp_path / "missing.gguf"), "my-model")
    assert result.success is False
    assert "GGUF file not found" in result.error


def test_create_ollama_model_success_with_system_prompt(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x" * 1024)

    with patch("subprocess.run", return_value=MagicMock(returncode=0)) as mock_run:
        result = create_ollama_model(str(gguf), "my-model", system_prompt='Be "helpful"')

    assert result.success is True
    assert result.ollama_model == "my-model"
    modelfile = gguf.parent / "Modelfile"
    content = modelfile.read_text()
    assert 'SYSTEM "Be \\"helpful\\""' in content
    mock_run.assert_called_once()


def test_create_ollama_model_success_without_system_prompt(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x" * 1024)

    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        result = create_ollama_model(str(gguf), "my-model")

    assert result.success is True
    modelfile = gguf.parent / "Modelfile"
    assert "SYSTEM" not in modelfile.read_text()


def test_create_ollama_model_create_fails(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")

    with patch("subprocess.run", return_value=MagicMock(returncode=1, stderr="bad modelfile\n")):
        result = create_ollama_model(str(gguf), "my-model")

    assert result.success is False
    assert "bad modelfile" in result.error


def test_create_ollama_model_ollama_not_found(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")

    with patch("subprocess.run", side_effect=FileNotFoundError()):
        result = create_ollama_model(str(gguf), "my-model")

    assert result.success is False
    assert "ollama not found" in result.error


def test_create_ollama_model_timeout(tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ollama", timeout=300)):
        result = create_ollama_model(str(gguf), "my-model")

    assert result.success is False
    assert "timed out" in result.error
