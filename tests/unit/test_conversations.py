"""Tests for the conversation history persistence system."""


import pytest

from llmstack.gateway.conversations import ConversationStore, Conversation, Message


@pytest.fixture
def store(tmp_path):
    return ConversationStore(db_path=tmp_path / "test_conv.db")


class TestConversationStore:
    def test_create_conversation(self, store):
        conv = store.create_conversation(title="Test Chat", model="llama3.2")
        assert conv.title == "Test Chat"
        assert conv.model == "llama3.2"
        assert len(conv.id) > 0

    def test_get_conversation(self, store):
        created = store.create_conversation(title="Lookup")
        fetched = store.get_conversation(created.id)
        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.title == "Lookup"

    def test_get_nonexistent(self, store):
        assert store.get_conversation("nope") is None

    def test_add_message(self, store):
        conv = store.create_conversation(title="Msg Test")
        msg = store.add_message(conv.id, role="user", content="Hello!", tokens=5)
        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.tokens == 5

        updated = store.get_conversation(conv.id)
        assert updated.message_count == 1
        assert updated.total_tokens == 5

    def test_get_messages(self, store):
        conv = store.create_conversation()
        store.add_message(conv.id, role="user", content="Hi")
        store.add_message(conv.id, role="assistant", content="Hello!")
        store.add_message(conv.id, role="user", content="How are you?")

        messages = store.get_messages(conv.id)
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    def test_get_messages_pagination(self, store):
        conv = store.create_conversation()
        for i in range(10):
            store.add_message(conv.id, role="user", content=f"msg {i}")

        page1 = store.get_messages(conv.id, limit=5, offset=0)
        page2 = store.get_messages(conv.id, limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].content != page2[0].content

    def test_list_conversations(self, store):
        store.create_conversation(title="First")
        store.create_conversation(title="Second")
        store.create_conversation(title="Third")

        convs = store.list_conversations()
        assert len(convs) == 3

    def test_list_with_limit(self, store):
        for i in range(5):
            store.create_conversation(title=f"Conv {i}")
        convs = store.list_conversations(limit=3)
        assert len(convs) == 3

    def test_search_conversations(self, store):
        conv = store.create_conversation(title="Python Discussion")
        store.add_message(conv.id, role="user", content="Tell me about Python")

        store.create_conversation(title="Go Discussion")

        results = store.list_conversations(search="Python")
        assert len(results) >= 1
        assert any(c.title == "Python Discussion" for c in results)

    def test_delete_conversation(self, store):
        conv = store.create_conversation(title="Delete Me")
        store.add_message(conv.id, role="user", content="bye")

        assert store.delete_conversation(conv.id) is True
        assert store.get_conversation(conv.id) is None
        assert store.get_messages(conv.id) == []

    def test_delete_nonexistent(self, store):
        assert store.delete_conversation("nope") is False

    def test_stats(self, store):
        conv = store.create_conversation()
        store.add_message(conv.id, role="user", content="hi", tokens=3)
        store.add_message(conv.id, role="assistant", content="hello", tokens=5)

        stats = store.get_stats()
        assert stats["total_conversations"] == 1
        assert stats["total_messages"] == 2
        assert stats["total_tokens"] == 8

    def test_tags(self, store):
        conv = store.create_conversation(tags=["python", "coding"])
        fetched = store.get_conversation(conv.id)
        assert fetched.tags == ["python", "coding"]

    def test_message_to_dict(self):
        msg = Message(role="user", content="test", model="llama3.2", tokens=10)
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["tokens"] == 10

    def test_conversation_to_dict(self):
        conv = Conversation(title="Test", model="llama3.2")
        d = conv.to_dict()
        assert d["title"] == "Test"
        assert "id" in d
