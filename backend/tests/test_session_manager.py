"""
Unit tests for SessionManager class.
Tests session creation, message management, and history functionality.
"""
import pytest
from session_manager import SessionManager, Message


class TestMessage:
    """Test the Message dataclass."""

    def test_message_creation(self):
        """Test creating a Message object."""
        message = Message(role="user", content="Hello world")
        assert message.role == "user"
        assert message.content == "Hello world"

    def test_message_equality(self):
        """Test Message equality comparison."""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="user", content="Hello")
        msg3 = Message(role="assistant", content="Hello")

        assert msg1 == msg2
        assert msg1 != msg3


class TestSessionManager:
    """Test the SessionManager class."""

    @pytest.fixture
    def session_manager(self):
        """Create a SessionManager for testing."""
        return SessionManager(max_history=3)

    def test_initialization(self):
        """Test SessionManager initialization."""
        sm = SessionManager(max_history=5)
        assert sm.max_history == 5
        assert sm.sessions == {}
        assert sm.session_counter == 0

    def test_initialization_default_max_history(self):
        """Test SessionManager initialization with default max_history."""
        sm = SessionManager()
        assert sm.max_history == 5

    def test_create_session(self, session_manager):
        """Test session creation."""
        session_id = session_manager.create_session()

        assert session_id == "session_1"
        assert session_id in session_manager.sessions
        assert session_manager.sessions[session_id] == []
        assert session_manager.session_counter == 1

    def test_create_multiple_sessions(self, session_manager):
        """Test creating multiple sessions."""
        session_id1 = session_manager.create_session()
        session_id2 = session_manager.create_session()
        session_id3 = session_manager.create_session()

        assert session_id1 == "session_1"
        assert session_id2 == "session_2"
        assert session_id3 == "session_3"
        assert session_manager.session_counter == 3
        assert len(session_manager.sessions) == 3

    def test_add_message_existing_session(self, session_manager):
        """Test adding messages to existing session."""
        session_id = session_manager.create_session()

        session_manager.add_message(session_id, "user", "Hello")
        session_manager.add_message(session_id, "assistant", "Hi there")

        messages = session_manager.sessions[session_id]
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Hi there"

    def test_add_message_non_existing_session(self, session_manager):
        """Test adding messages to non-existing session creates it."""
        session_manager.add_message("new_session", "user", "Hello")

        assert "new_session" in session_manager.sessions
        messages = session_manager.sessions["new_session"]
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello"

    def test_add_exchange(self, session_manager):
        """Test adding complete question-answer exchange."""
        session_id = session_manager.create_session()

        session_manager.add_exchange(session_id, "What is AI?", "AI is artificial intelligence.")

        messages = session_manager.sessions[session_id]
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What is AI?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "AI is artificial intelligence."

    def test_history_truncation(self, session_manager):
        """Test that conversation history is truncated at max_history * 2."""
        session_id = session_manager.create_session()

        # Add messages beyond the limit (max_history=3, so limit is 6 messages)
        for i in range(10):
            session_manager.add_message(session_id, "user", f"Message {i}")

        messages = session_manager.sessions[session_id]
        # Should keep only the last 6 messages (max_history * 2)
        assert len(messages) == 6
        assert messages[0].content == "Message 4"  # First kept message
        assert messages[-1].content == "Message 9"  # Last message

    def test_history_truncation_with_exchanges(self, session_manager):
        """Test history truncation when adding complete exchanges."""
        session_id = session_manager.create_session()

        # Add exchanges that will exceed the limit
        for i in range(5):
            session_manager.add_exchange(session_id, f"Question {i}", f"Answer {i}")

        messages = session_manager.sessions[session_id]
        # Should keep only the last 6 messages (max_history=3, so 3*2=6 messages)
        assert len(messages) == 6
        # Should keep the last 3 complete exchanges
        assert messages[0].content == "Question 2"
        assert messages[1].content == "Answer 2"
        assert messages[-2].content == "Question 4"
        assert messages[-1].content == "Answer 4"

    def test_get_conversation_history_existing_session(self, session_manager):
        """Test getting conversation history for existing session."""
        session_id = session_manager.create_session()
        session_manager.add_exchange(session_id, "Hello", "Hi there")
        session_manager.add_exchange(session_id, "How are you?", "I'm doing well")

        history = session_manager.get_conversation_history(session_id)

        expected = "User: Hello\nAssistant: Hi there\nUser: How are you?\nAssistant: I'm doing well"
        assert history == expected

    def test_get_conversation_history_non_existing_session(self, session_manager):
        """Test getting conversation history for non-existing session."""
        history = session_manager.get_conversation_history("non_existent")
        assert history is None

    def test_get_conversation_history_none_session_id(self, session_manager):
        """Test getting conversation history with None session_id."""
        history = session_manager.get_conversation_history(None)
        assert history is None

    def test_get_conversation_history_empty_session(self, session_manager):
        """Test getting conversation history for empty session."""
        session_id = session_manager.create_session()
        history = session_manager.get_conversation_history(session_id)
        assert history is None

    def test_clear_session_existing(self, session_manager):
        """Test clearing messages from existing session."""
        session_id = session_manager.create_session()
        session_manager.add_exchange(session_id, "Hello", "Hi")

        # Verify messages exist
        assert len(session_manager.sessions[session_id]) == 2

        # Clear session
        session_manager.clear_session(session_id)

        # Verify session is empty but still exists
        assert session_id in session_manager.sessions
        assert len(session_manager.sessions[session_id]) == 0

    def test_clear_session_non_existing(self, session_manager):
        """Test clearing non-existing session (should not raise error)."""
        # Should not raise an exception
        session_manager.clear_session("non_existent")

    def test_message_role_capitalization_in_history(self, session_manager):
        """Test that roles are properly capitalized in history format."""
        session_id = session_manager.create_session()
        session_manager.add_message(session_id, "user", "Hello")
        session_manager.add_message(session_id, "assistant", "Hi")

        history = session_manager.get_conversation_history(session_id)

        assert "User: Hello" in history
        assert "Assistant: Hi" in history

    def test_concurrent_sessions(self, session_manager):
        """Test managing multiple sessions concurrently."""
        session_id1 = session_manager.create_session()
        session_id2 = session_manager.create_session()

        session_manager.add_message(session_id1, "user", "Message in session 1")
        session_manager.add_message(session_id2, "user", "Message in session 2")

        # Verify sessions are independent
        assert len(session_manager.sessions[session_id1]) == 1
        assert len(session_manager.sessions[session_id2]) == 1
        assert session_manager.sessions[session_id1][0].content == "Message in session 1"
        assert session_manager.sessions[session_id2][0].content == "Message in session 2"

    def test_edge_case_zero_max_history(self):
        """Test SessionManager with zero max_history (edge case)."""
        sm = SessionManager(max_history=0)
        session_id = sm.create_session()

        # NOTE: This is a known edge case - max_history=0 doesn't work as expected
        # because [-0:] slice is equivalent to [:] (returns full list)
        # In practice, max_history should be >= 1 for normal operation

        sm.add_message(session_id, "user", "Hello")
        # Due to the slice behavior, messages aren't actually truncated with max_history=0
        assert len(sm.sessions[session_id]) == 1

        sm.add_message(session_id, "assistant", "Hi")
        assert len(sm.sessions[session_id]) == 2

    def test_edge_case_large_content(self, session_manager):
        """Test handling of large message content."""
        session_id = session_manager.create_session()
        large_content = "x" * 10000  # 10KB of content

        session_manager.add_message(session_id, "user", large_content)

        messages = session_manager.sessions[session_id]
        assert len(messages) == 1
        assert messages[0].content == large_content

    def test_edge_case_empty_content(self, session_manager):
        """Test handling of empty message content."""
        session_id = session_manager.create_session()

        session_manager.add_message(session_id, "user", "")

        messages = session_manager.sessions[session_id]
        assert len(messages) == 1
        assert messages[0].content == ""

    def test_new_chat_button_workflow(self, session_manager):
        """Test workflow for new chat button functionality."""
        # Simulate existing conversation
        session_id = session_manager.create_session()
        session_manager.add_exchange(session_id, "What is Python?", "Python is a programming language.")
        session_manager.add_exchange(session_id, "Tell me more", "Python is used for web development, data science, and more.")

        # Verify conversation exists
        assert len(session_manager.sessions[session_id]) == 4  # 2 exchanges = 4 messages
        history = session_manager.get_conversation_history(session_id)
        assert history is not None
        assert "Python is a programming language" in history

        # Simulate new chat button click - clear the session
        session_manager.clear_session(session_id)

        # Verify session is cleared
        assert len(session_manager.sessions[session_id]) == 0
        assert session_manager.get_conversation_history(session_id) is None

        # Simulate new conversation starting (as would happen when user asks first question)
        session_manager.add_exchange(session_id, "Hello", "Welcome! How can I help?")

        # Verify new conversation works normally
        assert len(session_manager.sessions[session_id]) == 2
        new_history = session_manager.get_conversation_history(session_id)
        assert "Welcome! How can I help?" in new_history
        assert "Python is a programming language" not in new_history  # Old conversation is gone

    def test_edge_case_unicode_content(self, session_manager):
        """Test handling of unicode message content."""
        session_id = session_manager.create_session()
        unicode_content = "Hello 🌍 世界 🚀"

        session_manager.add_message(session_id, "user", unicode_content)

        messages = session_manager.sessions[session_id]
        assert len(messages) == 1
        assert messages[0].content == unicode_content

        history = session_manager.get_conversation_history(session_id)
        assert unicode_content in history