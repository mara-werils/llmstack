"""Observe API routes — traces, quality, alerts, A/B tests."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/observe/traces")
async def list_traces(limit: int = 50, model: str = "", provider: str = ""):
    from llmstack.observe._state import get_trace_store

    store = get_trace_store()
    if store is None:
        return JSONResponse(content={"traces": [], "error": "observe not enabled"})

    kwargs = {"limit": limit}
    if model:
        kwargs["model"] = model
    if provider:
        kwargs["provider"] = provider

    traces = store.query(**kwargs)
    return JSONResponse(
        content={
            "traces": [t.to_dict() for t in traces],
            "total": store.total_count,
        }
    )


@router.get("/observe/traces/summary")
async def traces_summary():
    from llmstack.observe._state import get_trace_store

    store = get_trace_store()
    if store is None:
        return JSONResponse(content={"error": "observe not enabled"})
    return JSONResponse(content=store.summary())


@router.get("/observe/quality")
async def quality_summary():
    from llmstack.observe._state import get_tracker

    tracker = get_tracker()
    if tracker is None:
        return JSONResponse(content={"error": "observe not enabled"})
    return JSONResponse(content=tracker.summary())


@router.get("/observe/alerts")
async def list_alerts(limit: int = 20):
    from llmstack.observe._state import get_tracker

    tracker = get_tracker()
    if tracker is None:
        return JSONResponse(content={"alerts": []})
    alerts = tracker.get_alerts(limit=limit)
    return JSONResponse(content={"alerts": [a.to_dict() for a in alerts]})


@router.post("/observe/ab-test")
async def create_ab_test(request_data: dict):
    from llmstack.observe._state import get_ab_manager
    from llmstack.observe.ab_testing import ABTest

    manager = get_ab_manager()
    if manager is None:
        return JSONResponse(content={"error": "observe not enabled"}, status_code=503)

    test = ABTest(
        name=request_data.get("name", ""),
        model_a=request_data.get("model_a", ""),
        model_b=request_data.get("model_b", ""),
        traffic_split=request_data.get("traffic_split", 0.5),
    )
    manager.create_test(test)
    return JSONResponse(content={"status": "created", "test": test.name})


@router.get("/observe/ab-test")
async def list_ab_tests():
    from llmstack.observe._state import get_ab_manager

    manager = get_ab_manager()
    if manager is None:
        return JSONResponse(content={"tests": []})

    tests = manager.list_tests()
    results = []
    for t in tests:
        r = manager.get_results(t.name)
        results.append(r.to_dict() if r else {"name": t.name})
    return JSONResponse(content={"tests": results})


@router.get("/observe/ab-test/{test_name}")
async def get_ab_test(test_name: str):
    from llmstack.observe._state import get_ab_manager

    manager = get_ab_manager()
    if manager is None:
        return JSONResponse(content={"error": "observe not enabled"}, status_code=503)

    result = manager.get_results(test_name)
    if result is None:
        return JSONResponse(content={"error": f"test '{test_name}' not found"}, status_code=404)
    return JSONResponse(content=result.to_dict())


@router.get("/observe/stats")
async def observe_stats():
    """Aggregated observability stats for dashboard and CLI."""
    from llmstack.observe._state import get_trace_store, get_tracker, get_ab_manager

    result: dict = {}

    store = get_trace_store()
    if store is not None:
        result["traces"] = {
            "total": store.total_count,
            "summary": store.summary(),
        }

    tracker = get_tracker()
    if tracker is not None:
        alerts = tracker.get_alerts(limit=5)
        result["quality"] = {
            **tracker.summary(),
            "recent_alerts": [a.to_dict() for a in alerts],
        }

    manager = get_ab_manager()
    if manager is not None:
        tests = manager.list_tests()
        result["ab_tests"] = {"active": len(tests)}

    if not result:
        return JSONResponse(content={"error": "observe not enabled"}, status_code=503)

    return JSONResponse(content=result)


@router.delete("/observe/ab-test/{test_name}")
async def stop_ab_test(test_name: str):
    from llmstack.observe._state import get_ab_manager

    manager = get_ab_manager()
    if manager is None:
        return JSONResponse(content={"error": "observe not enabled"}, status_code=503)
    manager.stop_test(test_name)
    return JSONResponse(content={"status": "stopped", "test": test_name})
