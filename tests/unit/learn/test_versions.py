"""Tests for model version management (llmstack.learn.versions)."""

from __future__ import annotations

import json
import time

import pytest

from llmstack.learn.store import FeedbackStore
from llmstack.learn.versions import ModelVersion, ModelVersionManager


@pytest.fixture
def store(tmp_path):
    s = FeedbackStore(db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def version_mgr(store, tmp_path):
    return ModelVersionManager(store=store, versions_dir=tmp_path / "versions")


class TestModelVersion:
    def test_display_name_active(self):
        mv = ModelVersion(
            version="3",
            base_model="base",
            quality_score=0.875,
            is_active=True,
            timestamp=time.time(),
        )
        name = mv.display_name
        assert name.startswith("v3 (")
        assert "quality=0.875" in name
        assert "[active]" in name

    def test_display_name_inactive(self):
        mv = ModelVersion(
            version="1",
            base_model="base",
            quality_score=0.5,
            is_active=False,
            timestamp=time.time(),
        )
        name = mv.display_name
        assert name.startswith("v1 (")
        assert "quality=0.500" in name
        assert "[active]" not in name

    def test_defaults(self):
        mv = ModelVersion(version="1", base_model="base")
        assert mv.adapter_path == ""
        assert mv.quality_score == 0.0
        assert mv.is_active is False
        assert mv.train_run_id == 0
        assert mv.metadata == {}
        assert mv.timestamp > 0


class TestCountAndHas:
    def test_version_count_empty(self, version_mgr):
        assert version_mgr.version_count == 0

    def test_version_count_after_create(self, version_mgr):
        version_mgr.create_version(base_model="b", adapter_path="")
        version_mgr.create_version(base_model="b", adapter_path="")
        assert version_mgr.version_count == 2

    def test_has_versions_false(self, version_mgr):
        assert version_mgr.has_versions() is False

    def test_has_versions_true(self, version_mgr):
        version_mgr.create_version(base_model="b", adapter_path="")
        assert version_mgr.has_versions() is True


class TestCreateVersion:
    def test_create_basic(self, version_mgr):
        mv = version_mgr.create_version(
            base_model="my-model",
            adapter_path="",
            train_run_id=7,
            quality_score=0.9,
            activate=True,
            metadata={"foo": "bar"},
        )
        assert isinstance(mv, ModelVersion)
        assert mv.version == "1"
        assert mv.base_model == "my-model"
        assert mv.train_run_id == 7
        assert mv.quality_score == 0.9
        assert mv.is_active is True
        assert mv.metadata == {"foo": "bar"}

    def test_create_writes_metadata_file(self, version_mgr):
        mv = version_mgr.create_version(base_model="m", adapter_path="")
        meta_path = version_mgr.versions_dir / mv.version / "version.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert data["version"] == "1"
        assert data["base_model"] == "m"

    def test_create_none_metadata(self, version_mgr):
        mv = version_mgr.create_version(base_model="m", adapter_path="", metadata=None)
        assert mv.metadata == {}

    def test_create_copies_file_adapter(self, version_mgr, tmp_path):
        adapter_file = tmp_path / "adapter.bin"
        adapter_file.write_text("weights")
        mv = version_mgr.create_version(base_model="m", adapter_path=str(adapter_file))
        dest = version_mgr.versions_dir / mv.version / "adapter"
        assert dest.exists()
        assert dest.is_file()
        assert dest.read_text() == "weights"
        assert mv.adapter_path == str(dest)

    def test_create_copies_dir_adapter(self, version_mgr, tmp_path):
        adapter_dir = tmp_path / "adapter_dir"
        adapter_dir.mkdir()
        (adapter_dir / "config.json").write_text("{}")
        (adapter_dir / "weights.bin").write_text("w")
        mv = version_mgr.create_version(base_model="m", adapter_path=str(adapter_dir))
        dest = version_mgr.versions_dir / mv.version / "adapter"
        assert dest.is_dir()
        assert (dest / "config.json").exists()
        assert (dest / "weights.bin").exists()
        assert mv.adapter_path == str(dest)

    def test_create_nonexistent_adapter_keeps_path(self, version_mgr, tmp_path):
        missing = str(tmp_path / "does_not_exist")
        mv = version_mgr.create_version(base_model="m", adapter_path=missing)
        assert mv.adapter_path == missing

    def test_create_not_active(self, version_mgr):
        mv = version_mgr.create_version(base_model="m", adapter_path="", activate=False)
        assert mv.is_active is False
        assert version_mgr.get_active() is None


class TestGetActive:
    def test_no_active(self, version_mgr):
        assert version_mgr.get_active() is None

    def test_get_active(self, version_mgr):
        version_mgr.create_version(
            base_model="m",
            adapter_path="",
            quality_score=0.7,
            activate=True,
            metadata={"a": 1},
        )
        active = version_mgr.get_active()
        assert active is not None
        assert active.version == "1"
        assert active.is_active is True
        assert active.quality_score == 0.7
        assert active.metadata == {"a": 1}

    def test_active_follows_latest(self, version_mgr):
        version_mgr.create_version(base_model="m", adapter_path="", activate=True)
        version_mgr.create_version(base_model="m", adapter_path="", activate=True)
        active = version_mgr.get_active()
        assert active.version == "2"


class TestActivate:
    def test_activate_existing(self, version_mgr):
        version_mgr.create_version(base_model="m", adapter_path="", activate=True)
        version_mgr.create_version(base_model="m", adapter_path="", activate=True)
        # version 2 is active; activate version 1
        assert version_mgr.activate("1") is True
        assert version_mgr.get_active().version == "1"

    def test_activate_not_found(self, version_mgr):
        version_mgr.create_version(base_model="m", adapter_path="", activate=True)
        assert version_mgr.activate("999") is False

    def test_activate_not_found_empty(self, version_mgr):
        assert version_mgr.activate("1") is False


class TestRollback:
    def test_rollback_no_versions(self, version_mgr):
        assert version_mgr.rollback() is None

    def test_rollback_single_version(self, version_mgr):
        version_mgr.create_version(base_model="m", adapter_path="", activate=True)
        assert version_mgr.rollback() is None

    def test_rollback_to_previous(self, version_mgr):
        version_mgr.create_version(
            base_model="m", adapter_path="", quality_score=0.7, activate=True
        )
        version_mgr.create_version(
            base_model="m", adapter_path="", quality_score=0.8, activate=True
        )
        # v2 active, v1 previous; rollback should activate v1
        rolled = version_mgr.rollback()
        assert rolled is not None
        assert rolled.version == "1"
        assert rolled.is_active is True
        assert version_mgr.get_active().version == "1"

    def test_rollback_fallback_no_active(self, version_mgr, store):
        # Two versions, none active -> previous stays None -> fallback to versions[1]
        store.add_model_version(version="1", base_model="m", is_active=False)
        store.add_model_version(version="2", base_model="m", is_active=False)
        rolled = version_mgr.rollback()
        assert rolled is not None
        # versions ordered by timestamp DESC -> [v2, v1]; fallback uses index 1 (v1)
        assert rolled.version == "1"


class TestListVersions:
    def test_list_empty(self, version_mgr):
        assert version_mgr.list_versions() == []

    def test_list_returns_modelversions(self, version_mgr):
        version_mgr.create_version(
            base_model="m", adapter_path="", quality_score=0.6, metadata={"k": "v"}
        )
        version_mgr.create_version(base_model="m", adapter_path="")
        versions = version_mgr.list_versions()
        assert len(versions) == 2
        assert all(isinstance(v, ModelVersion) for v in versions)
        # most recent first
        assert versions[0].version == "2"
        v1 = next(v for v in versions if v.version == "1")
        assert v1.quality_score == 0.6
        assert v1.metadata == {"k": "v"}

    def test_list_respects_limit(self, version_mgr):
        for _ in range(3):
            version_mgr.create_version(base_model="m", adapter_path="")
        assert len(version_mgr.list_versions(limit=2)) == 2


class TestCompare:
    def test_compare_no_metrics(self, version_mgr):
        result = version_mgr.compare("1", "2")
        assert result["version_a"] == "1"
        assert result["version_b"] == "2"
        assert result["score_a"] == 0.0
        assert result["score_b"] == 0.0
        assert result["improvement"] == 0.0
        # tie -> defaults to version_a
        assert result["better"] == "1"

    def test_compare_b_better(self, version_mgr, store):
        store.add_quality_snapshot("1", "overall", 0.6)
        store.add_quality_snapshot("2", "overall", 0.9)
        result = version_mgr.compare("1", "2")
        assert result["score_a"] == 0.6
        assert result["score_b"] == 0.9
        assert result["improvement"] == pytest.approx(0.3)
        assert result["better"] == "2"

    def test_compare_a_better(self, version_mgr, store):
        store.add_quality_snapshot("1", "overall", 0.95)
        store.add_quality_snapshot("2", "overall", 0.5)
        result = version_mgr.compare("1", "2")
        assert result["improvement"] == pytest.approx(-0.45)
        assert result["better"] == "1"


class TestNextVersion:
    def test_first_version(self, version_mgr):
        assert version_mgr._next_version() == "1"

    def test_increments(self, version_mgr):
        version_mgr.create_version(base_model="m", adapter_path="")
        assert version_mgr._next_version() == "2"

    def test_non_numeric_version_fallback(self, version_mgr, store):
        # Latest version is non-integer -> ValueError -> timestamp fallback
        store.add_model_version(version="abc", base_model="m")
        nxt = version_mgr._next_version()
        assert nxt.isdigit()
        # timestamp-based fallback is large
        assert int(nxt) > 1_000_000_000
