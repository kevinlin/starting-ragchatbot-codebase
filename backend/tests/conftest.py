"""
Shared test configuration and fixtures for the RAG chatbot backend.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any, List

from models import Course, Lesson, CourseChunk


@pytest.fixture
def sample_course():
    """Create a sample course for testing."""
    return Course(
        title="Test Course",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Introduction",
                lesson_link="https://example.com/lesson/1"
            ),
            Lesson(
                lesson_number=2,
                title="Advanced Topics",
                lesson_link="https://example.com/lesson/2"
            )
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing."""
    return [
        CourseChunk(
            content="This is the first chunk of content.",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="This is the second chunk of content.",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="This is content from lesson 2.",
            course_title="Test Course",
            lesson_number=2,
            chunk_index=2
        )
    ]


@pytest.fixture
def sample_course_document():
    """Sample course document content for testing."""
    return """Course Title: Test Course
Course Link: https://example.com/course
Course Instructor: Test Instructor

Lesson 1: Introduction
Lesson Link: https://example.com/lesson/1
This is the introduction lesson content. It covers basic concepts and provides an overview of the course structure.

Lesson 2: Advanced Topics
Lesson Link: https://example.com/lesson/2
This lesson covers more advanced topics. It builds on the foundation from the first lesson.
"""


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_course_file(temp_dir, sample_course_document):
    """Create a temporary course file for testing."""
    file_path = Path(temp_dir) / "test_course.txt"
    file_path.write_text(sample_course_document)
    return str(file_path)


@pytest.fixture
def mock_chroma_client():
    """Create a mock ChromaDB client for testing."""
    client = Mock()
    collection = Mock()

    # Setup collection methods
    collection.query.return_value = {
        "documents": [["Sample document content"]],
        "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
        "distances": [[0.5]]
    }
    collection.get.return_value = {
        "ids": ["test_id"],
        "metadatas": [{"title": "Test Course", "instructor": "Test Instructor"}]
    }
    collection.add = Mock()

    # Setup client methods
    client.get_or_create_collection.return_value = collection
    client.delete_collection = Mock()

    return client


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing."""
    client = Mock()

    # Mock response object
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test response from Claude."
    mock_response.stop_reason = "end_turn"

    client.messages.create.return_value = mock_response

    return client


@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic response with tool use for testing."""
    response = Mock()
    response.stop_reason = "tool_use"

    # Mock tool use content block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.input = {"query": "test query"}
    tool_block.id = "tool_use_123"

    # Mock text content block
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "I need to search for information."

    response.content = [text_block, tool_block]

    return response


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    from vector_store import SearchResults

    return SearchResults(
        documents=["Document 1", "Document 2"],
        metadata=[
            {"course_title": "Test Course", "lesson_number": 1},
            {"course_title": "Test Course", "lesson_number": 2}
        ],
        distances=[0.3, 0.7]
    )


@pytest.fixture
def mock_embedding_function():
    """Mock embedding function for testing."""
    mock_func = Mock()
    mock_func.__call__ = Mock(return_value=[[0.1, 0.2, 0.3]])
    return mock_func


class MockChromaEmbeddingFunction:
    """Mock ChromaDB embedding function for testing."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, texts: List[str]) -> List[List[float]]:
        # Return mock embeddings based on text length
        return [[0.1 * len(text), 0.2, 0.3] for text in texts]


@pytest.fixture
def mock_sentence_transformers(monkeypatch):
    """Mock sentence transformers to avoid downloading models during tests."""
    mock_transformer = Mock()
    mock_transformer.encode.return_value = [[0.1, 0.2, 0.3]]

    def mock_sentence_transformer_init(*args, **kwargs):
        return mock_transformer

    monkeypatch.setattr(
        "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction",
        MockChromaEmbeddingFunction
    )

    return mock_transformer