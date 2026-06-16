"""Unit tests for llmstack.sdk.batch BatchProcessor.run / run_sync."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from llmstack.sdk.batch import (
    BatchItem,
    BatchProcessor,
    BatchResult,
    BatchSummary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code=200, content="hello", total_tokens=7):
    """Build a MagicMock httpx-style response object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "usage": {"total_tokens": total_tokens},
    }
    return resp


def _patch_async_client(post_side_effect=None, post_return=None):
    """Return a patch context manager for httpx.AsyncClient.

    The patched AsyncClient is an async context manager whose .post is an
    AsyncMock configured with the given side effect / return value.
    """
    mock_client = MagicMock()
    mock_post = AsyncMock()
    if post_side_effect is not None:
        mock_post.side_effect = post_side_effect
    else:
        mock_post.return_value = post_return
    mock_client.post = mock_post

    # AsyncClient(...) used as `async with` context manager.
    async_cm = MagicMock()
    async_cm.__aenter__ = AsyncMock(return_value=mock_client)
    async_cm.__aexit__ = AsyncMock(return_value=False)

    factory = MagicMock(return_value=async_cm)
    return patch("httpx.AsyncClient", factory), mock_post


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRunSuccess:
    async def test_single_item_success(self):
        proc = BatchProcessor()
        items = [BatchItem(id=1, prompt="Hello")]
        cm, mock_post = _patch_async_client(post_return=_make_response())
        with cm:
            summary = await proc.run(items)

        assert isinstance(summary, BatchSummary)
        assert summary.total == 1
        assert summary.completed == 1
        assert summary.failed == 0
        assert summary.total_tokens == 7
        assert len(summary.results) == 1
        r = summary.results[0]
        assert isinstance(r, BatchResult)
        assert r.success is True
        assert r.response == "hello"
        assert r.tokens_used == 7
        assert r.error is None
        assert r.id == 1

    async def test_multiple_items_success(self):
        proc = BatchProcessor(concurrency=2)
        items = [
            BatchItem(id=1, prompt="a"),
            BatchItem(id=2, prompt="b"),
            BatchItem(id=3, prompt="c"),
        ]
        cm, mock_post = _patch_async_client(
            post_return=_make_response(content="ok", total_tokens=5)
        )
        with cm:
            summary = await proc.run(items)

        assert summary.total == 3
        assert summary.completed == 3
        assert summary.failed == 0
        assert summary.total_tokens == 15
        assert mock_post.call_count == 3
        # avg_duration uses total/len(items)
        assert summary.avg_duration == summary.total_duration / 3

    async def test_payload_includes_system_prompt(self):
        proc = BatchProcessor(model="default-model")
        items = [
            BatchItem(
                id="x",
                prompt="user text",
                system_prompt="be terse",
                model="custom-model",
                max_tokens=256,
                temperature=0.1,
            )
        ]
        cm, mock_post = _patch_async_client(post_return=_make_response())
        with cm:
            await proc.run(items)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "custom-model"
        assert payload["max_tokens"] == 256
        assert payload["temperature"] == 0.1
        assert payload["messages"][0] == {"role": "system", "content": "be terse"}
        assert payload["messages"][1] == {"role": "user", "content": "user text"}

    async def test_payload_without_system_prompt(self):
        proc = BatchProcessor()
        items = [BatchItem(id=1, prompt="just user")]
        cm, mock_post = _patch_async_client(post_return=_make_response())
        with cm:
            await proc.run(items)

        payload = mock_post.call_args.kwargs["json"]
        # No system message — model falls back to default.
        assert payload["model"] == "llama3.2"
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"

    async def test_default_model_used_when_item_model_none(self):
        proc = BatchProcessor(model="my-default")
        items = [BatchItem(id=1, prompt="p")]
        cm, mock_post = _patch_async_client(post_return=_make_response())
        with cm:
            await proc.run(items)

        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "my-default"

    async def test_missing_choices_yields_empty_response(self):
        proc = BatchProcessor()
        items = [BatchItem(id=1, prompt="p")]
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {}  # no choices, no usage
        cm, _ = _patch_async_client(post_return=resp)
        with cm:
            summary = await proc.run(items)

        r = summary.results[0]
        assert r.success is True
        assert r.response == ""
        assert r.tokens_used == 0


# ---------------------------------------------------------------------------
# Auth header behavior
# ---------------------------------------------------------------------------


class TestHeaders:
    async def test_authorization_header_with_api_key(self):
        proc = BatchProcessor(api_key="sk-secret")
        items = [BatchItem(id=1, prompt="p")]
        cm, mock_post = _patch_async_client(post_return=_make_response())
        with cm:
            await proc.run(items)

        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-secret"
        assert headers["Content-Type"] == "application/json"

    async def test_no_authorization_header_without_api_key(self):
        proc = BatchProcessor()
        items = [BatchItem(id=1, prompt="p")]
        cm, mock_post = _patch_async_client(post_return=_make_response())
        with cm:
            await proc.run(items)

        headers = mock_post.call_args.kwargs["headers"]
        assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Error / edge branches
# ---------------------------------------------------------------------------


class TestRunErrors:
    async def test_non_200_status_marks_failed(self):
        proc = BatchProcessor()
        items = [BatchItem(id=1, prompt="p")]
        resp = MagicMock()
        resp.status_code = 500
        cm, _ = _patch_async_client(post_return=resp)
        with cm:
            summary = await proc.run(items)

        assert summary.completed == 0
        assert summary.failed == 1
        r = summary.results[0]
        assert r.success is False
        assert r.error == "HTTP 500"
        assert r.response == ""
        assert r.tokens_used == 0

    async def test_exception_during_request_marks_failed(self):
        proc = BatchProcessor()
        items = [BatchItem(id=1, prompt="p")]
        cm, _ = _patch_async_client(post_side_effect=RuntimeError("boom"))
        with cm:
            summary = await proc.run(items)

        assert summary.completed == 0
        assert summary.failed == 1
        r = summary.results[0]
        assert r.success is False
        assert r.error == "boom"

    async def test_mixed_success_and_failure(self):
        proc = BatchProcessor(concurrency=3)
        items = [
            BatchItem(id=1, prompt="ok"),
            BatchItem(id=2, prompt="bad"),
        ]

        good = _make_response(content="yes", total_tokens=4)
        bad = MagicMock()
        bad.status_code = 429

        cm, _ = _patch_async_client(post_side_effect=[good, bad])
        with cm:
            summary = await proc.run(items)

        assert summary.total == 2
        assert summary.completed == 1
        assert summary.failed == 1
        assert summary.total_tokens == 4
        by_id = {r.id: r for r in summary.results}
        assert by_id[1].success is True
        assert by_id[2].success is False
        assert by_id[2].error == "HTTP 429"


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


class TestEmpty:
    async def test_empty_items(self):
        proc = BatchProcessor()
        cm, mock_post = _patch_async_client(post_return=_make_response())
        with cm:
            summary = await proc.run([])

        assert summary.total == 0
        assert summary.completed == 0
        assert summary.failed == 0
        assert summary.total_tokens == 0
        assert summary.results == []
        # avg_duration must not divide by zero (max(1, 0)).
        assert summary.avg_duration == summary.total_duration
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    async def test_callback_invoked_per_item(self):
        proc = BatchProcessor(concurrency=1)
        calls = []
        proc.on_progress(lambda completed, total, result: calls.append((completed, total, result)))

        items = [BatchItem(id=1, prompt="a"), BatchItem(id=2, prompt="b")]
        cm, _ = _patch_async_client(post_return=_make_response())
        with cm:
            summary = await proc.run(items)

        assert len(calls) == 2
        # total is always len(items)
        assert all(c[1] == 2 for c in calls)
        # completed counter reaches the total.
        assert {c[0] for c in calls} == {1, 2}
        # results passed are BatchResult instances.
        assert all(isinstance(c[2], BatchResult) for c in calls)
        assert summary.completed == 2

    async def test_no_callback_does_not_error(self):
        proc = BatchProcessor()
        items = [BatchItem(id=1, prompt="a")]
        cm, _ = _patch_async_client(post_return=_make_response())
        with cm:
            summary = await proc.run(items)
        assert summary.completed == 1


# ---------------------------------------------------------------------------
# run_sync wrapper (line 167)
# ---------------------------------------------------------------------------


class TestRunSync:
    def test_run_sync_delegates_to_run(self):
        proc = BatchProcessor()
        items = [BatchItem(id=1, prompt="hello")]
        cm, _ = _patch_async_client(post_return=_make_response())
        with cm:
            summary = proc.run_sync(items)

        assert isinstance(summary, BatchSummary)
        assert summary.total == 1
        assert summary.completed == 1
        assert summary.results[0].response == "hello"

    def test_run_sync_empty(self):
        proc = BatchProcessor()
        summary = proc.run_sync([])
        assert summary.total == 0
        assert summary.results == []
