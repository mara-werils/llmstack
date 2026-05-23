"""Tests for multi-armed bandit model selection."""

from __future__ import annotations

import pytest

from llmstack.learn.bandit import BanditConfig, ModelBandit


@pytest.fixture
def bandit():
    return ModelBandit(models=["small", "medium", "large"])


class TestModelBandit:
    def test_select_returns_valid_model(self, bandit):
        model = bandit.select()
        assert model in ["small", "medium", "large"]

    def test_explores_all_arms_initially(self, bandit):
        seen = set()
        for _ in range(20):
            model = bandit.select()
            bandit.update(model, reward=0.5)
            seen.add(model)
        assert len(seen) == 3

    def test_update_tracks_stats(self, bandit):
        bandit.update("small", reward=0.8)
        bandit.update("small", reward=0.9)
        stats = bandit.get_stats()
        assert stats["arms"]["small"]["pulls"] == 2
        assert stats["arms"]["small"]["successes"] == 2

    def test_negative_reward_tracks_failures(self, bandit):
        bandit.update("large", reward=0.2)
        stats = bandit.get_stats()
        assert stats["arms"]["large"]["failures"] == 1
        assert stats["arms"]["large"]["successes"] == 0

    def test_converges_to_best_arm(self):
        bandit = ModelBandit(
            models=["good", "bad"],
            config=BanditConfig(min_pulls_per_arm=3),
        )
        # Train heavily
        for _ in range(50):
            bandit.update("good", reward=0.9)
            bandit.update("bad", reward=0.1)

        # Should mostly select "good"
        selections = [bandit.select() for _ in range(20)]
        good_count = selections.count("good")
        assert good_count > 15

    def test_category_tracking(self, bandit):
        bandit.update("small", reward=0.9, category="code")
        bandit.update("large", reward=0.9, category="chat")

        code_stats = bandit.get_stats(category="code")
        assert code_stats["arms"]["small"]["pulls"] == 1

    def test_ucb1_strategy(self):
        bandit = ModelBandit(
            models=["a", "b"],
            config=BanditConfig(strategy="ucb1", min_pulls_per_arm=2),
        )
        for _ in range(10):
            model = bandit.select()
            bandit.update(model, reward=0.5)
        assert bandit.get_stats()["total_pulls"] == 10

    def test_epsilon_greedy_strategy(self):
        bandit = ModelBandit(
            models=["a", "b"],
            config=BanditConfig(strategy="epsilon_greedy", epsilon=0.1, min_pulls_per_arm=2),
        )
        for _ in range(10):
            model = bandit.select()
            bandit.update(model, reward=0.5)
        assert bandit.get_stats()["total_pulls"] == 10

    def test_stats_output(self, bandit):
        bandit.update("small", reward=0.8)
        stats = bandit.get_stats()
        assert "total_pulls" in stats
        assert "strategy" in stats
        assert "arms" in stats
        assert "best_model" in stats

    def test_mean_reward(self, bandit):
        bandit.update("medium", reward=0.6)
        bandit.update("medium", reward=0.8)
        stats = bandit.get_stats()
        assert abs(stats["arms"]["medium"]["mean_reward"] - 0.7) < 0.01
