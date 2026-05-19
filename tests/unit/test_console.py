"""Tests for console display helpers."""

from llmstack.cli.console import console, banner, success, failure, warn, info


def test_console_exists():
    assert console is not None


def test_banner_no_crash(capsys):
    banner("Test Banner", "subtitle")


def test_success_no_crash(capsys):
    success("it works")


def test_failure_no_crash(capsys):
    failure("it broke")


def test_warn_no_crash(capsys):
    warn("be careful")


def test_info_no_crash(capsys):
    info("fyi")


def test_spinner_context_manager():
    from llmstack.cli.console import spinner
    with spinner("working...") as p:
        assert p is not None


def test_timer_context_manager(capsys):
    from llmstack.cli.console import timer
    with timer("test"):
        pass
