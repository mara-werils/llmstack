"""Smoke tests for the `llmstack benchmark` command (mock path)."""

from __future__ import annotations

from llmstack.cli.commands.benchmark import benchmark


def test_mock_run_prints_report(capsys) -> None:
    benchmark(mock=True, warmup=0)
    out = capsys.readouterr().out
    assert "llmstack benchmark" in out
    assert "Cost vs cloud" in out
    assert "provably local" in out


def test_mock_run_writes_files(tmp_path, capsys) -> None:
    out_md = tmp_path / "report.md"
    benchmark(mock=True, warmup=0, output=str(out_md))
    assert out_md.exists()
    assert out_md.with_suffix(".json").exists()
    assert "Wrote" in capsys.readouterr().out


def test_unknown_suite_is_handled(capsys) -> None:
    benchmark(mock=True, suite_name="does-not-exist")
    assert "Unknown suite" in capsys.readouterr().out


def test_baseline_flows_through(capsys) -> None:
    benchmark(mock=True, warmup=0, baseline="gpt-4o")
    out = capsys.readouterr().out
    assert "GPT-4o" in out


def test_no_proof_mode(capsys) -> None:
    benchmark(mock=True, warmup=0, proof=False)
    out = capsys.readouterr().out
    # Without a proof we don't print the "provably local" confirmation.
    assert "provably local" not in out


def test_ollama_failure_is_friendly(capsys, monkeypatch) -> None:
    # Force the non-mock path to fail fast without a real Ollama.
    import llmstack.cli.commands.benchmark as bench_mod

    def _boom(model, ollama_url):
        def generate(prompt):
            raise RuntimeError("connection refused")

        return generate

    monkeypatch.setattr(bench_mod, "_ollama_generator", _boom)
    benchmark(mock=False, warmup=0)
    out = capsys.readouterr().out
    # Per-task failures are caught by the runner, so the run completes with no
    # measurements; the command should report that clearly rather than crash.
    assert "Every task failed" in out
