"""SDK client for the learning pipeline API.

Provides programmatic access to submit feedback, query learning status,
and manage model versions through the gateway's /learn endpoints.
"""

from __future__ import annotations

from typing import Any

import httpx


class LearnClient:
    """Client for the learning pipeline REST API.

    Usage:
        from llmstack.sdk import Client

        client = Client()
        learn = client.learn

        # Submit feedback
        learn.thumbs_up(query="How do I X?", response="Do Y.")
        learn.correct(query="How?", response="bad", correction="good")

        # Check status
        status = learn.status()
        print(status["metrics"])

        # Trigger training
        learn.train()
    """

    def __init__(self, base_url: str, headers: dict[str, str]):
        self._base_url = base_url.rstrip("/")
        self._headers = headers

    def thumbs_up(
        self,
        query: str = "",
        response: str = "",
        model: str = "",
    ) -> dict[str, Any]:
        """Submit positive feedback."""
        return self._submit_feedback("thumbs_up", query=query, response=response, model=model)

    def thumbs_down(
        self,
        query: str = "",
        response: str = "",
        model: str = "",
    ) -> dict[str, Any]:
        """Submit negative feedback."""
        return self._submit_feedback("thumbs_down", query=query, response=response, model=model)

    def correct(
        self,
        query: str,
        response: str,
        correction: str,
        model: str = "",
    ) -> dict[str, Any]:
        """Submit a correction."""
        return self._submit_feedback(
            "correction",
            query=query,
            response=response,
            correction=correction,
            model=model,
        )

    def prefer(
        self,
        query: str,
        chosen: str,
        rejected: str,
        model: str = "",
    ) -> dict[str, Any]:
        """Submit a preference (A over B)."""
        return self._submit_feedback(
            "preference",
            query=query,
            response=rejected,
            correction=chosen,
            model=model,
        )

    def status(self) -> dict[str, Any]:
        """Get learning pipeline status."""
        resp = httpx.get(
            f"{self._base_url}/learn/status",
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def versions(self) -> dict[str, Any]:
        """List model versions."""
        resp = httpx.get(
            f"{self._base_url}/learn/versions",
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def preferences(self) -> dict[str, Any]:
        """Get learned user preferences."""
        resp = httpx.get(
            f"{self._base_url}/learn/preferences",
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def train(self) -> dict[str, Any]:
        """Trigger a training run."""
        resp = httpx.post(
            f"{self._base_url}/learn/train",
            headers=self._headers,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def rollback(self) -> dict[str, Any]:
        """Rollback to previous model version."""
        resp = httpx.post(
            f"{self._base_url}/learn/rollback",
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _submit_feedback(
        self,
        feedback_type: str,
        query: str = "",
        response: str = "",
        correction: str = "",
        model: str = "",
    ) -> dict[str, Any]:
        """Submit feedback to the learning endpoint."""
        resp = httpx.post(
            f"{self._base_url}/learn/feedback",
            headers=self._headers,
            json={
                "feedback_type": feedback_type,
                "query": query,
                "response": response,
                "correction": correction,
                "model": model,
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()
