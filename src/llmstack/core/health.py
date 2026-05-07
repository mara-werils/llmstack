"""Health check polling for services."""

from __future__ import annotations

import asyncio

import httpx


async def wait_healthy(url: str, timeout: int = 120, interval: float = 2.0) -> bool:
    """Poll a health endpoint until it returns 200 or timeout expires."""
    elapsed = 0.0
    async with httpx.AsyncClient(timeout=5) as client:
        while elapsed < timeout:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
                pass
            await asyncio.sleep(interval)
            elapsed += interval
    return False
