"""Tests for multi-armed bandit model selection."""

from __future__ import annotations

from unittest.mock import patch

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

    def test_min_exploration_is_balanced(self):
        bandit = ModelBandit(
            models=["a", "b", "c"],
            config=BanditConfig(min_pulls_per_arm=2),
        )
        # Pull "a" once; "b"/"c" still have zero pulls. The exploration phase
        # must prefer a least-pulled arm, not return "a" again just because it
        # comes first in insertion order.
        bandit.update("a", reward=0.5)
        assert bandit.select() in ("b", "c")

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

    def test_best_arm_none_when_no_pulls(self, bandit):
        assert bandit.best_arm is None

    def test_best_arm_none_when_empty(self):
        bandit = ModelBandit(models=[])
        assert bandit.best_arm is None

    def test_best_arm_returns_highest_reward(self, bandit):
        bandit.update("small", reward=0.2)
        bandit.update("medium", reward=0.9)
        assert bandit.best_arm == "medium"

    def test_total_pulls(self, bandit):
        assert bandit.total_pulls == 0
        bandit.update("small", reward=0.5)
        bandit.update("medium", reward=0.5)
        assert bandit.total_pulls == 2

    def test_unknown_strategy_falls_back_to_thompson(self):
        bandit = ModelBandit(
            models=["a", "b"],
            config=BanditConfig(strategy="not-a-real-strategy", min_pulls_per_arm=0),
        )
        model = bandit.select()
        assert model in ("a", "b")

    def test_ucb1_select_with_zero_total_pulls(self, bandit):
        arms = bandit._get_arms("fresh-category")
        assert bandit._ucb1_select(arms) in arms

    def test_ucb1_select_picks_unpulled_arm_when_others_pulled(self, bandit):
        arms = bandit._get_arms("mixed")
        arms["small"].pulls = 5
        arms["small"].total_reward = 4.0
        # "medium" and "large" remain unpulled -> selected immediately.
        assert bandit._ucb1_select(arms) in ("medium", "large")

    def test_epsilon_greedy_explores_when_random_below_epsilon(self, bandit):
        arms = bandit._get_arms("explore")
        with patch("llmstack.learn.bandit.random.random", return_value=0.0):
            with patch("llmstack.learn.bandit.random.choice", return_value="large") as mock_choice:
                result = bandit._epsilon_greedy_select(arms)
        mock_choice.assert_called_once()
        assert result == "large"
