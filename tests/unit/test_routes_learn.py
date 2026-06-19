"""Tests for the /learn API routes."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from llmstack.gateway.routes import learn as learn_route


# --- Fakes for the lazily-imported collaborators ---


class _FakeStore:
    """Stand-in for llmstack.learn.store.FeedbackStore."""

    last_instance = None

    def __init__(self, *args, **kwargs):
        self.added = []
        self.closed = False
        self.unused_count = 0
        self.marked_used = []
        type(self).last_instance = self

    def add_feedback(self, feedback):
        self.added.append(feedback)

    def get_unused_feedback_count(self):
        return self.unused_count

    def mark_feedback_used(self, ids):
        self.marked_used.append(ids)

    def close(self):
        self.closed = True


class _FakeLearner:
    """Stand-in for llmstack.learn.preferences.PreferenceLearner."""

    last_instance = None

    def __init__(self, *args, store=None, **kwargs):
        self.store = store
        self.learned = []
        self.profile = {"tone": "concise"}
        self.additions = "Be concise."
        type(self).last_instance = self

    def learn_from_feedback(self, feedback):
        self.learned.append(feedback)

    def get_profile(self):
        return self.profile

    def get_system_prompt_additions(self):
        return self.additions


class _FakeAnalytics:
    """Stand-in for llmstack.learn.analytics.LearningAnalytics."""

    summary = {
        "status": "healthy",
        "metrics": {"total_feedback": 3},
        "recommendations": ["keep going"],
    }

    def __init__(self, *args, **kwargs):
        pass

    def get_summary(self):
        return self.summary


class _FakeVersion:
    def __init__(self, version, active=False):
        self.version = version
        self.base_model = "llama3.2"
        self.quality_score = 0.9
        self.is_active = active
        self.timestamp = 1234.0


class _FakeVersionManager:
    """Stand-in for llmstack.learn.versions.ModelVersionManager."""

    versions = None
    active = None
    rollback_result = None

    def __init__(self, *args, **kwargs):
        pass

    def list_versions(self):
        return self.versions if self.versions is not None else []

    def get_active(self):
        return self.active

    def rollback(self):
        return self.rollback_result


class _FakeDataset:
    def __init__(self, total=2, sft=1, dpo=1, feedback_ids=None):
        self._total = total
        self.sft_examples = list(range(sft))
        self.dpo_examples = list(range(dpo))
        self.feedback_ids = feedback_ids or ["fb1", "fb2"]
        self.saved_to = None

    @property
    def total_examples(self):
        return self._total

    def save(self, output_dir):
        self.saved_to = output_dir
        return output_dir / "dataset.jsonl"


class _FakeDatasetGenerator:
    """Stand-in for llmstack.learn.dataset.DatasetGenerator."""

    dataset = None

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, strategy=None):
        return self.dataset


@pytest.fixture
def client(monkeypatch):
    # The route imports these lazily from their source modules, so patch there.
    monkeypatch.setattr("llmstack.learn.store.FeedbackStore", _FakeStore)
    monkeypatch.setattr("llmstack.learn.preferences.PreferenceLearner", _FakeLearner)
    monkeypatch.setattr("llmstack.learn.analytics.LearningAnalytics", _FakeAnalytics)
    monkeypatch.setattr("llmstack.learn.versions.ModelVersionManager", _FakeVersionManager)
    monkeypatch.setattr("llmstack.learn.dataset.DatasetGenerator", _FakeDatasetGenerator)

    _FakeStore.last_instance = None
    _FakeLearner.last_instance = None
    _FakeVersionManager.versions = None
    _FakeVersionManager.active = None
    _FakeVersionManager.rollback_result = None
    _FakeDatasetGenerator.dataset = None

    app = FastAPI()
    app.include_router(learn_route.router)
    return TestClient(app)


# --- /feedback ---


class TestFeedback:
    def test_thumbs_up_no_correction(self, client):
        _FakeStore.last_instance = None
        resp = client.post(
            "/learn/feedback",
            json={"feedback_type": "thumbs_up", "query": "q", "response": "r"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["pending_count"] == 0
        assert body["id"]
        # store was used and closed; learner NOT invoked (no correction)
        store = _FakeStore.last_instance
        assert store.closed is True
        assert len(store.added) == 1
        assert _FakeLearner.last_instance is None

    def test_correction_triggers_preference_learner(self, client):
        resp = client.post(
            "/learn/feedback",
            json={
                "feedback_type": "correction",
                "query": "q",
                "response": "bad",
                "correction": "good",
            },
        )
        assert resp.status_code == 200
        # has_correction is True -> learner instantiated and used
        learner = _FakeLearner.last_instance
        assert learner is not None
        assert len(learner.learned) == 1

    def test_pending_count_passthrough(self, client):
        # Make the store report a non-zero unused count.
        def _make_store(*a, **k):
            s = _FakeStore()
            s.unused_count = 5
            _FakeStore.last_instance = s
            return s

        import llmstack.learn.store as store_mod

        store_mod.FeedbackStore = _make_store  # type: ignore[assignment]
        resp = client.post("/learn/feedback", json={"feedback_type": "thumbs_down"})
        assert resp.status_code == 200
        assert resp.json()["pending_count"] == 5

    def test_invalid_feedback_type_400(self, client):
        resp = client.post("/learn/feedback", json={"feedback_type": "not_a_real_type"})
        assert resp.status_code == 400
        assert "Invalid feedback_type" in resp.json()["detail"]

    def test_missing_feedback_type_422(self, client):
        # feedback_type is required by the pydantic model.
        resp = client.post("/learn/feedback", json={})
        assert resp.status_code == 422


# --- /status ---


class TestStatus:
    def test_status(self, client):
        resp = client.get("/learn/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["metrics"] == {"total_feedback": 3}
        assert body["recommendations"] == ["keep going"]
        assert _FakeStore.last_instance.closed is True

    def test_status_missing_recommendations_defaults_empty(self, client, monkeypatch):
        monkeypatch.setattr(
            _FakeAnalytics,
            "summary",
            {"status": "ok", "metrics": {}},
        )
        resp = client.get("/learn/status")
        assert resp.status_code == 200
        assert resp.json()["recommendations"] == []


# --- /versions ---


class TestVersions:
    def test_versions_empty(self, client):
        resp = client.get("/learn/versions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["versions"] == []
        assert body["active_version"] is None

    def test_versions_with_active(self, client):
        active = _FakeVersion("v2", active=True)
        _FakeVersionManager.versions = [_FakeVersion("v1"), active]
        _FakeVersionManager.active = active
        resp = client.get("/learn/versions")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["versions"]) == 2
        assert body["versions"][0]["version"] == "v1"
        assert body["versions"][0]["base_model"] == "llama3.2"
        assert body["versions"][1]["is_active"] is True
        assert body["active_version"] == "v2"


# --- /preferences ---


class TestPreferences:
    def test_preferences(self, client):
        resp = client.get("/learn/preferences")
        assert resp.status_code == 200
        body = resp.json()
        assert body["preferences"] == {"tone": "concise"}
        assert body["system_prompt_additions"] == "Be concise."
        assert _FakeStore.last_instance.closed is True


# --- /train ---


class TestTrain:
    def test_train_no_unused_feedback(self, client):
        # default unused count is 0 -> early return
        resp = client.post("/learn/train")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "No unused feedback" in body["message"]
        assert _FakeStore.last_instance.closed is True

    def test_train_no_examples_generated(self, client):
        def _make_store(*a, **k):
            s = _FakeStore()
            s.unused_count = 4
            _FakeStore.last_instance = s
            return s

        import llmstack.learn.store as store_mod

        store_mod.FeedbackStore = _make_store  # type: ignore[assignment]
        _FakeDatasetGenerator.dataset = _FakeDataset(total=0, sft=0, dpo=0)

        resp = client.post("/learn/train")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "No training examples" in body["message"]

    def test_train_success(self, client):
        def _make_store(*a, **k):
            s = _FakeStore()
            s.unused_count = 4
            _FakeStore.last_instance = s
            return s

        import llmstack.learn.store as store_mod

        store_mod.FeedbackStore = _make_store  # type: ignore[assignment]
        ds = _FakeDataset(total=3, sft=2, dpo=1, feedback_ids=["a", "b"])
        _FakeDatasetGenerator.dataset = ds

        resp = client.post("/learn/train")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert "3 examples" in body["message"]
        assert body["details"]["sft_examples"] == 2
        assert body["details"]["dpo_examples"] == 1
        assert body["details"]["dataset_path"].endswith("dataset.jsonl")
        # feedback ids were marked used
        store = _FakeStore.last_instance
        assert store.marked_used == [["a", "b"]]
        assert store.closed is True


# --- /rollback ---


class TestRollback:
    def test_rollback_success(self, client):
        _FakeVersionManager.rollback_result = _FakeVersion("v1")
        resp = client.post("/learn/rollback")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["version"] == "v1"
        assert body["quality"] == 0.9
        assert _FakeStore.last_instance.closed is True

    def test_rollback_no_previous_version(self, client):
        _FakeVersionManager.rollback_result = None
        resp = client.post("/learn/rollback")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert "No previous version" in body["error"]
