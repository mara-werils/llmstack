"""Tests for CLI error handler."""

from __future__ import annotations
import pytest
from llmstack.cli.errors import friendly_errors


class TestFriendlyErrors:
    def test_keyboard_interrupt(self):
        @friendly_errors
        def boom():
            raise KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            boom()
        assert exc_info.value.code == 130

    def test_system_exit_passthrough(self):
        @friendly_errors
        def exit_fn():
            raise SystemExit(0)

        with pytest.raises(SystemExit) as exc_info:
            exit_fn()
        assert exc_info.value.code == 0

    def test_connection_refused(self):
        @friendly_errors
        def fail():
            raise ConnectionError("Connection refused")

        with pytest.raises(SystemExit):
            fail()
