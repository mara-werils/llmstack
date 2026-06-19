"""Comprehensive tests for the SDK LearnClient (llmstack.sdk.learn_client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from llmstack.sdk.learn_client import LearnClient

BASE = "http://localhost:8000"
HEADERS = {"Authorization": "Bearer sk-test", "Content-Type": "application/json"}


def _make_client(base_url: str = BASE + "/") -> LearnClient:
    return LearnClient(base_url=base_url, headers=dict(HEADERS))


def _ok_response(payload: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {} if payload is None else payload
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_strips_trailing_slash(self) -> None:
        c = LearnClient(base_url="http://h:8000/", headers={})
        assert c._base_url == "http://h:8000"

    def test_strips_multiple_trailing_slashes(self) -> None:
        c = LearnClient(base_url="http://h:8000///", headers={})
        assert c._base_url == "http://h:8000"

    def test_no_trailing_slash_unchanged(self) -> None:
        c = LearnClient(base_url="http://h:8000", headers={})
        assert c._base_url == "http://h:8000"

    def test_headers_stored(self) -> None:
        hdrs = {"Authorization": "Bearer abc"}
        c = LearnClient(base_url=BASE, headers=hdrs)
        assert c._headers is hdrs


# ---------------------------------------------------------------------------
# Feedback helpers (POST /learn/feedback)
# ---------------------------------------------------------------------------


class TestThumbsUp:
    def test_returns_json(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_ok_response({"status": "queued"})) as post:
            result = c.thumbs_up(query="How do I X?", response="Do Y.")
        assert result == {"status": "queued"}
        url = post.call_args[0][0]
        assert url == f"{BASE}/learn/feedback"

    def test_payload_and_headers(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_ok_response()) as post:
            c.thumbs_up(query="q", response="r", model="m1")
        kwargs = post.call_args[1]
        assert kwargs["headers"] == HEADERS
        assert kwargs["timeout"] == 10
        assert kwargs["json"] == {
            "feedback_type": "thumbs_up",
            "query": "q",
            "response": "r",
            "correction": "",
            "model": "m1",
        }

    def test_defaults_empty_strings(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_ok_response()) as post:
            c.thumbs_up()
        payload = post.call_args[1]["json"]
        assert payload["query"] == ""
        assert payload["response"] == ""
        assert payload["model"] == ""
        assert payload["feedback_type"] == "thumbs_up"


class TestThumbsDown:
    def test_feedback_type(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_ok_response({"ok": True})) as post:
            result = c.thumbs_down(query="q", response="r", model="m")
        assert result == {"ok": True}
        assert post.call_args[1]["json"]["feedback_type"] == "thumbs_down"


class TestCorrect:
    def test_payload(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_ok_response()) as post:
            c.correct(query="How?", response="bad", correction="good", model="m")
        payload = post.call_args[1]["json"]
        assert payload == {
            "feedback_type": "correction",
            "query": "How?",
            "response": "bad",
            "correction": "good",
            "model": "m",
        }

    def test_model_defaults_empty(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_ok_response()) as post:
            c.correct(query="q", response="r", correction="c")
        assert post.call_args[1]["json"]["model"] == ""


class TestPrefer:
    def test_maps_chosen_to_correction_and_rejected_to_response(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_ok_response()) as post:
            c.prefer(query="q", chosen="A", rejected="B", model="m")
        payload = post.call_args[1]["json"]
        assert payload["feedback_type"] == "preference"
        assert payload["response"] == "B"  # rejected
        assert payload["correction"] == "A"  # chosen
        assert payload["query"] == "q"
        assert payload["model"] == "m"

    def test_returns_json(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_ok_response({"id": 1})):
            assert c.prefer(query="q", chosen="A", rejected="B") == {"id": 1}


# ---------------------------------------------------------------------------
# GET endpoints
# ---------------------------------------------------------------------------


class TestStatus:
    def test_url_and_result(self) -> None:
        c = _make_client()
        payload = {"metrics": {"pending": 3}}
        with patch("httpx.get", return_value=_ok_response(payload)) as get:
            result = c.status()
        assert result == payload
        assert get.call_args[0][0] == f"{BASE}/learn/status"
        assert get.call_args[1]["headers"] == HEADERS
        assert get.call_args[1]["timeout"] == 10


class TestVersions:
    def test_url_and_result(self) -> None:
        c = _make_client()
        payload = {"versions": ["v1", "v2"]}
        with patch("httpx.get", return_value=_ok_response(payload)) as get:
            result = c.versions()
        assert result == payload
        assert get.call_args[0][0] == f"{BASE}/learn/versions"


class TestPreferences:
    def test_url_and_result(self) -> None:
        c = _make_client()
        payload = {"preferences": {"tone": "concise"}}
        with patch("httpx.get", return_value=_ok_response(payload)) as get:
            result = c.preferences()
        assert result == payload
        assert get.call_args[0][0] == f"{BASE}/learn/preferences"


# ---------------------------------------------------------------------------
# POST endpoints (train / rollback)
# ---------------------------------------------------------------------------


class TestTrain:
    def test_url_timeout_and_result(self) -> None:
        c = _make_client()
        payload = {"run_id": "abc"}
        with patch("httpx.post", return_value=_ok_response(payload)) as post:
            result = c.train()
        assert result == payload
        assert post.call_args[0][0] == f"{BASE}/learn/train"
        assert post.call_args[1]["timeout"] == 300
        assert post.call_args[1]["headers"] == HEADERS

    def test_no_json_body(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_ok_response()) as post:
            c.train()
        assert "json" not in post.call_args[1]


class TestRollback:
    def test_url_timeout_and_result(self) -> None:
        c = _make_client()
        payload = {"rolled_back_to": "v1"}
        with patch("httpx.post", return_value=_ok_response(payload)) as post:
            result = c.rollback()
        assert result == payload
        assert post.call_args[0][0] == f"{BASE}/learn/rollback"
        assert post.call_args[1]["timeout"] == 10


# ---------------------------------------------------------------------------
# Error branches: raise_for_status propagation
# ---------------------------------------------------------------------------


def _error_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    request = httpx.Request("GET", BASE)
    http_resp = httpx.Response(status_code, request=request)
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"{status_code} error", request=request, response=http_resp
    )
    return resp


class TestErrorBranches:
    def test_status_raises_on_http_error(self) -> None:
        c = _make_client()
        with patch("httpx.get", return_value=_error_response(500)):
            with pytest.raises(httpx.HTTPStatusError):
                c.status()

    def test_versions_raises_on_http_error(self) -> None:
        c = _make_client()
        with patch("httpx.get", return_value=_error_response(404)):
            with pytest.raises(httpx.HTTPStatusError):
                c.versions()

    def test_preferences_raises_on_http_error(self) -> None:
        c = _make_client()
        with patch("httpx.get", return_value=_error_response(403)):
            with pytest.raises(httpx.HTTPStatusError):
                c.preferences()

    def test_train_raises_on_http_error(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_error_response(500)):
            with pytest.raises(httpx.HTTPStatusError):
                c.train()

    def test_rollback_raises_on_http_error(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_error_response(409)):
            with pytest.raises(httpx.HTTPStatusError):
                c.rollback()

    def test_feedback_raises_on_http_error(self) -> None:
        c = _make_client()
        with patch("httpx.post", return_value=_error_response(400)):
            with pytest.raises(httpx.HTTPStatusError):
                c.thumbs_up(query="q", response="r")

    def test_request_error_propagates(self) -> None:
        c = _make_client()
        with patch("httpx.get", side_effect=httpx.ConnectError("boom")):
            with pytest.raises(httpx.ConnectError):
                c.status()


# ---------------------------------------------------------------------------
# Integration-style via MockTransport (real httpx request building)
# ---------------------------------------------------------------------------


class TestWithMockTransport:
    def test_feedback_roundtrip_via_transport(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["method"] = request.method
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={"status": "stored"})

        transport = httpx.MockTransport(handler)

        c = _make_client()

        # Patch httpx.post to route through a client backed by MockTransport.
        def fake_post(url, **kwargs):
            with httpx.Client(transport=transport) as client:
                return client.post(url, **kwargs)

        with patch("httpx.post", side_effect=fake_post):
            result = c.correct(query="q", response="bad", correction="good")

        assert result == {"status": "stored"}
        assert captured["url"] == f"{BASE}/learn/feedback"
        assert captured["method"] == "POST"
        assert captured["auth"] == "Bearer sk-test"

    def test_status_roundtrip_via_transport(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == f"{BASE}/learn/status"
            return httpx.Response(200, json={"metrics": {"ok": 1}})

        transport = httpx.MockTransport(handler)

        c = _make_client()

        def fake_get(url, **kwargs):
            with httpx.Client(transport=transport) as client:
                return client.get(url, **kwargs)

        with patch("httpx.get", side_effect=fake_get):
            result = c.status()

        assert result == {"metrics": {"ok": 1}}

    def test_http_error_via_transport(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        transport = httpx.MockTransport(handler)
        c = _make_client()

        def fake_post(url, **kwargs):
            with httpx.Client(transport=transport) as client:
                return client.post(url, **kwargs)

        with patch("httpx.post", side_effect=fake_post):
            with pytest.raises(httpx.HTTPStatusError):
                c.train()
