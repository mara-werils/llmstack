"""Conversation history API routes — list, search, and manage chat history."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from llmstack.gateway.conversations import ConversationStore

router = APIRouter(tags=["Conversations"])

_store: ConversationStore | None = None


def get_store() -> ConversationStore:
    global _store
    if _store is None:
        _store = ConversationStore()
    return _store


class CreateConversationRequest(BaseModel):
    title: str = ""
    model: str = ""
    tags: list[str] = Field(default_factory=list)


class AddMessageRequest(BaseModel):
    role: str
    content: str
    model: str = ""
    tokens: int = 0
    latency_ms: float = 0.0


@router.get("/conversations")
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
):
    """List recent conversations with optional search."""
    store = get_store()
    convs = store.list_conversations(limit=limit, offset=offset, search=search)
    return {"conversations": [c.to_dict() for c in convs]}


@router.post("/conversations", status_code=201)
async def create_conversation(req: CreateConversationRequest):
    """Start a new conversation."""
    store = get_store()
    conv = store.create_conversation(
        title=req.title,
        model=req.model,
        tags=req.tags,
    )
    return conv.to_dict()


@router.get("/conversations/stats")
async def conversation_stats():
    """Get conversation statistics."""
    return get_store().get_stats()


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation details with messages."""
    store = get_store()
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = store.get_messages(conversation_id)
    result = conv.to_dict()
    result["messages"] = [m.to_dict() for m in messages]
    return result


@router.post("/conversations/{conversation_id}/messages")
async def add_message(conversation_id: str, req: AddMessageRequest):
    """Add a message to a conversation."""
    store = get_store()
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msg = store.add_message(
        conversation_id=conversation_id,
        role=req.role,
        content=req.content,
        model=req.model,
        tokens=req.tokens,
        latency_ms=req.latency_ms,
    )
    return msg.to_dict()


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    store = get_store()
    if not store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"deleted": True}
