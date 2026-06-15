"""Tests for the request replay store."""

from __future__ import annotations

import json

import pytest

from llmstack.gateway.replay import RecordedRequest, ReplayStore


@pytest.fixture
def store():
    return ReplayStore(max_size=100)


def _req(**kw) -> RecordedRequest:
    base = dict(model="llama3", provider="ollama", messages=[{"role": "user", "content": "hi"}])
    base.update(kw)
    return RecordedRequest(**base)


class TestRecordedRequest:
    def test_auto_id_and_timestamp(self):
        r = RecordedRequest()
        assert len(r.id) == 12
        assert r.timestamp > 0

    def test_explicit_id_preserved(self):
        r = RecordedRequest(id="fixed-id")
        assert r.id == "fixed-id"

    def test_total_tokens(self):
        r = _req(input_tokens=10, output_tokens=5)
        assert r.total_tokens == 15

    def test_is_error_via_error_string(self):
        assert _req(error="boom").is_error is True

    def test_is_error_via_status_code(self):
        assert _req(status_code=500).is_error is True

    def test_not_error_when_clean(self):
        assert _req(status_code=200).is_error is False

    def test_to_dict_roundtrip_fields(self):
        d = _req(latency_ms=12.345, cost_usd=0.0000123456).to_dict()
        assert d["model"] == "llama3"
        assert d["latency_ms"] == 12.3
        assert d["cost_usd"] == round(0.0000123456, 6)

    def test_to_replay_payload_includes_max_tokens(self):
        payload = _req(max_tokens=256, extra_params={"top_p": 0.9}).to_replay_payload()
        assert payload["max_tokens"] == 256
        assert payload["top_p"] == 0.9

    def test_to_replay_payload_omits_falsy_max_tokens(self):
        payload = _req(max_tokens=None).to_replay_payload()
        assert "max_tokens" not in payload


class TestReplayStore:
    def test_records_when_recording(self, store):
        store.record(_req())
        assert len(store.list_records()) == 1

    def test_does_not_record_when_stopped(self, store):
        store.stop_recording()
        assert store.is_recording is False
        store.record(_req())
        assert store.list_records() == []
        store.start_recording()
        assert store.is_recording is True

    def test_get_by_id(self, store):
        r = _req()
        store.record(r)
        assert store.get(r.id) is r
        assert store.get("missing") is None

    def test_list_filters(self, store):
        store.record(_req(model="a", provider="p1", tags=["x"]))
        store.record(_req(model="b", provider="p2", error="err"))
        assert len(store.list_records(model="a")) == 1
        assert len(store.list_records(provider="p2")) == 1
        assert len(store.list_records(tag="x")) == 1
        assert len(store.list_records(has_error=True)) == 1
        assert len(store.list_records(has_error=False)) == 1

    def test_list_respects_limit(self, store):
        for _ in range(5):
            store.record(_req())
        assert len(store.list_records(limit=2)) == 2

    def test_clear(self, store):
        store.record(_req())
        store.record(_req())
        assert store.clear() == 2
        assert store.list_records() == []

    def test_export_import_jsonl(self, store, tmp_path):
        store.record(_req(model="a", input_tokens=3, output_tokens=4))
        store.record(_req(model="b", error="bad"))
        path = tmp_path / "sub" / "records.jsonl"
        assert store.export_jsonl(path) == 2

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["model"] == "a"

        fresh = ReplayStore()
        assert fresh.import_jsonl(path) == 2
        assert len(fresh.list_records()) == 2

    def test_import_missing_file(self, store, tmp_path):
        assert store.import_jsonl(tmp_path / "nope.jsonl") == 0

    def test_import_skips_blank_lines(self, store, tmp_path):
        path = tmp_path / "r.jsonl"
        path.write_text(json.dumps(_req().to_dict()) + "\n\n")
        assert store.import_jsonl(path) == 1

    def test_stats_empty(self, store):
        stats = store.get_stats()
        assert stats == {"total": 0, "recording": True}

    def test_stats_aggregates(self, store):
        store.record(_req(model="a", cost_usd=0.01))
        store.record(_req(model="a", cost_usd=0.02, error="x"))
        store.record(_req(model="b", cost_usd=0.03))
        stats = store.get_stats()
        assert stats["total"] == 3
        assert stats["errors"] == 1
        assert stats["total_cost_usd"] == pytest.approx(0.06)
        assert stats["models"] == {"a": 2, "b": 1}

    def test_maxlen_eviction(self):
        small = ReplayStore(max_size=2)
        for i in range(3):
            small.record(_req(notes=str(i)))
        assert len(small.list_records()) == 2
