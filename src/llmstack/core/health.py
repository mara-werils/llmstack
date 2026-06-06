"""Health check polling for services."""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)


async def wait_healthy(url: str, timeout: int = 120, interval: float = 2.0) -> bool:
    """Poll a health endpoint until it returns 200 or timeout expires."""
    elapsed = 0.0
    attempts = 0
    async with httpx.AsyncClient(timeout=5) as client:
        while elapsed < timeout:
            attempts += 1
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    logger.debug("Health check passed for %s after %d attempts", url, attempts)
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(interval)
            elapsed += interval
    logger.warning(
        "Health check timed out for %s after %d attempts (%.0fs)", url, attempts, elapsed
    )
    return False
