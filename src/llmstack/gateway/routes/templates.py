"""Prompt template API routes — CRUD, render, search, and versioning."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llmstack.gateway.prompt_templates import TemplateStore, BUILTIN_TEMPLATES

router = APIRouter(tags=["Templates"])

_store: TemplateStore | None = None


def get_store() -> TemplateStore:
    global _store
    if _store is None:
        _store = TemplateStore()
        # Load built-in templates
        for bt in BUILTIN_TEMPLATES:
            try:
                _store.create(**bt)
            except ValueError:
                pass
    return _store


# --- Request / Response models ---


class CreateTemplateRequest(BaseModel):
    name: str
    content: str
    description: str = ""
    category: str = "general"
    tags: list[str] = Field(default_factory=list)


class UpdateTemplateRequest(BaseModel):
    content: str


class RenderTemplateRequest(BaseModel):
    variables: dict[str, str] = Field(default_factory=dict)


class RollbackRequest(BaseModel):
    version: int


# --- Routes ---


@router.get("/templates")
async def list_templates(
    category: str | None = None,
    tag: str | None = None,
    limit: int = 100,
):
    """List all prompt templates with optional filters."""
    store = get_store()
    templates = store.list_all(category=category, tag=tag, limit=limit)
    return {
        "templates": [t.to_dict() for t in templates],
        "total": store.count,
    }


@router.post("/templates", status_code=201)
async def create_template(req: CreateTemplateRequest):
    """Create a new prompt template."""
    store = get_store()
    try:
        template = store.create(
            name=req.name,
            content=req.content,
            description=req.description,
            category=req.category,
            tags=req.tags,
        )
        return template.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/templates/{name_or_id}")
async def get_template(name_or_id: str):
    """Get a specific template by name or ID."""
    store = get_store()
    template = store.get(name_or_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return template.to_dict()


@router.put("/templates/{name_or_id}")
async def update_template(name_or_id: str, req: UpdateTemplateRequest):
    """Update a template (creates a new version)."""
    store = get_store()
    template = store.update(name_or_id, content=req.content)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return template.to_dict()


@router.delete("/templates/{name_or_id}")
async def delete_template(name_or_id: str):
    """Delete a template."""
    store = get_store()
    if not store.delete(name_or_id):
        raise HTTPException(status_code=404, detail="Template not found")
    return {"deleted": True}


@router.post("/templates/{name_or_id}/render")
async def render_template(name_or_id: str, req: RenderTemplateRequest):
    """Render a template with variable substitution."""
    store = get_store()
    result = store.render(name_or_id, variables=req.variables)
    if result is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return {"rendered": result}


@router.post("/templates/{name_or_id}/rollback")
async def rollback_template(name_or_id: str, req: RollbackRequest):
    """Roll back a template to a specific version."""
    store = get_store()
    try:
        template = store.rollback(name_or_id, version=req.version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return template.to_dict()


@router.get("/templates/{name_or_id}/versions")
async def list_versions(name_or_id: str):
    """List all versions of a template."""
    store = get_store()
    template = store.get(name_or_id)
    if template is None:
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "name": template.name,
        "current_version": template.current_version,
        "versions": [
            {
                "version": v.version,
                "variables": v.variables,
                "created_at": v.created_at,
                "author": v.author,
                "content_preview": v.content[:200],
            }
            for v in template.versions
        ],
    }


@router.get("/templates/search/{query}")
async def search_templates(query: str, limit: int = 20):
    """Search templates by name, description, or tags."""
    store = get_store()
    results = store.search(query, limit=limit)
    return {
        "query": query,
        "results": [t.to_dict() for t in results],
        "total": len(results),
    }
