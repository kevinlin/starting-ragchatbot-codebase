"""
Unit tests for VectorStore class.
Tests vector storage, search functionality, and ChromaDB integration.
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from vector_store import VectorStore, SearchResults
from models import Course, CourseChunk


class TestSearchResults:
    """Test the SearchResults dataclass."""

    def test_search_results_creation(self):
        """Test creating SearchResults object."""
        results = SearchResults(
            documents=["doc1", "doc2"],
            metadata=[{"key": "value1"}, {"key": "value2"}],
            distances=[0.1, 0.9]
        )

        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"key": "value1"}, {"key": "value2"}]
        assert results.distances == [0.1, 0.9]
        assert results.error is None

    def test_search_results_from_chroma(self):
        """Test creating SearchResults from ChromaDB results."""
        chroma_results = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "value1"}, {"key": "value2"}]],
            "distances": [[0.1, 0.9]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"key": "value1"}, {"key": "value2"}]
        assert results.distances == [0.1, 0.9]

    def test_search_results_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results."""
        chroma_results = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_search_results_empty_with_error(self):
        """Test creating empty SearchResults with error."""
        results = SearchResults.empty("No results found")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "No results found"

    def test_search_results_is_empty(self):
        """Test is_empty method."""
        empty_results = SearchResults([], [], [])
        non_empty_results = SearchResults(["doc"], [{}], [0.5])

        assert empty_results.is_empty() is True
        assert non_empty_results.is_empty() is False


class TestVectorStore:
    """Test the VectorStore class."""

    @pytest.fixture
    def mock_chromadb(self):
        """Mock ChromaDB for testing."""
        with patch('vector_store.chromadb') as mock_chromadb:
            yield mock_chromadb

    @pytest.fixture
    def vector_store(self, mock_chromadb, mock_sentence_transformers):
        """Create a VectorStore for testing."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore(
            chroma_path="/tmp/test_chroma",
            embedding_model="test-model",
            max_results=5
        )
        store.client = mock_client
        store.course_catalog = mock_collection
        store.course_content = mock_collection
        return store

    def test_initialization(self, mock_chromadb, mock_sentence_transformers):
        """Test VectorStore initialization."""
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client

        store = VectorStore("/tmp/test", "test-model", max_results=10)

        assert store.max_results == 10
        mock_chromadb.PersistentClient.assert_called_once()

    def test_add_course_metadata(self, vector_store, sample_course):
        """Test adding course metadata to catalog."""
        vector_store.add_course_metadata(sample_course)

        # Verify add was called with correct parameters
        vector_store.course_catalog.add.assert_called_once()
        call_args = vector_store.course_catalog.add.call_args

        # Check documents
        assert call_args[1]["documents"] == [sample_course.title]

        # Check metadata structure
        metadata = call_args[1]["metadatas"][0]
        assert metadata["title"] == sample_course.title
        assert metadata["instructor"] == sample_course.instructor
        assert metadata["course_link"] == sample_course.course_link

        # Check lessons are serialized as JSON
        lessons_data = json.loads(metadata["lessons_json"])
        assert len(lessons_data) == 2
        assert lessons_data[0]["lesson_number"] == 1
        assert lessons_data[0]["lesson_title"] == "Introduction"

        # Check IDs
        assert call_args[1]["ids"] == [sample_course.title]

    def test_add_course_content(self, vector_store, sample_course_chunks):
        """Test adding course content chunks."""
        vector_store.add_course_content(sample_course_chunks)

        vector_store.course_content.add.assert_called_once()
        call_args = vector_store.course_content.add.call_args

        # Check documents
        expected_docs = [chunk.content for chunk in sample_course_chunks]
        assert call_args[1]["documents"] == expected_docs

        # Check metadata
        expected_metadata = [{
            "course_title": chunk.course_title,
            "lesson_number": chunk.lesson_number,
            "chunk_index": chunk.chunk_index
        } for chunk in sample_course_chunks]
        assert call_args[1]["metadatas"] == expected_metadata

        # Check IDs
        expected_ids = [
            f"{chunk.course_title.replace(' ', '_')}_{chunk.chunk_index}"
            for chunk in sample_course_chunks
        ]
        assert call_args[1]["ids"] == expected_ids

    def test_add_course_content_empty(self, vector_store):
        """Test adding empty course content list."""
        vector_store.add_course_content([])

        # Should not call add if empty
        vector_store.course_content.add.assert_not_called()

    def test_search_basic(self, vector_store, sample_search_results):
        """Test basic search functionality."""
        vector_store.course_content.query.return_value = {
            "documents": [sample_search_results.documents],
            "metadatas": [sample_search_results.metadata],
            "distances": [sample_search_results.distances]
        }

        results = vector_store.search("test query")

        assert results.documents == sample_search_results.documents
        assert results.metadata == sample_search_results.metadata
        assert results.distances == sample_search_results.distances

    def test_search_with_course_filter(self, vector_store):
        """Test search with course name filter."""
        # Mock course resolution
        vector_store.course_catalog.query.return_value = {
            "documents": [["Test Course"]],
            "metadatas": [[{"title": "Test Course"}]]
        }

        # Mock search results
        vector_store.course_content.query.return_value = {
            "documents": [["result"]],
            "metadatas": [[{"course_title": "Test Course"}]],
            "distances": [[0.5]]
        }

        results = vector_store.search("test query", course_name="Test")

        # Should resolve course name first
        vector_store.course_catalog.query.assert_called_once_with(
            query_texts=["Test"], n_results=1
        )

        # Should search with course filter
        vector_store.course_content.query.assert_called_once()
        call_args = vector_store.course_content.query.call_args
        assert call_args[1]["where"] == {"course_title": "Test Course"}

    def test_search_with_lesson_filter(self, vector_store):
        """Test search with lesson number filter."""
        vector_store.course_content.query.return_value = {
            "documents": [["result"]],
            "metadatas": [[{"lesson_number": 1}]],
            "distances": [[0.5]]
        }

        results = vector_store.search("test query", lesson_number=1)

        call_args = vector_store.course_content.query.call_args
        assert call_args[1]["where"] == {"lesson_number": 1}

    def test_search_with_course_and_lesson_filter(self, vector_store):
        """Test search with both course and lesson filters."""
        # Mock course resolution
        vector_store.course_catalog.query.return_value = {
            "documents": [["Test Course"]],
            "metadatas": [[{"title": "Test Course"}]]
        }

        vector_store.course_content.query.return_value = {
            "documents": [["result"]],
            "metadatas": [[{"course_title": "Test Course", "lesson_number": 1}]],
            "distances": [[0.5]]
        }

        results = vector_store.search("test query", course_name="Test", lesson_number=1)

        call_args = vector_store.course_content.query.call_args
        expected_filter = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 1}
            ]
        }
        assert call_args[1]["where"] == expected_filter

    def test_search_course_not_found(self, vector_store):
        """Test search when course name cannot be resolved."""
        # Mock empty course resolution
        vector_store.course_catalog.query.return_value = {
            "documents": [[]],
            "metadatas": [[]]
        }

        results = vector_store.search("test query", course_name="NonExistent")

        assert results.error == "No course found matching 'NonExistent'"
        assert results.is_empty()

    def test_search_exception_handling(self, vector_store):
        """Test search exception handling."""
        vector_store.course_content.query.side_effect = Exception("Database error")

        results = vector_store.search("test query")

        assert results.error == "Search error: Database error"
        assert results.is_empty()

    def test_resolve_course_name_success(self, vector_store):
        """Test successful course name resolution."""
        vector_store.course_catalog.query.return_value = {
            "documents": [["Test Course"]],
            "metadatas": [[{"title": "Resolved Course Name"}]]
        }

        resolved = vector_store._resolve_course_name("Test")

        assert resolved == "Resolved Course Name"

    def test_resolve_course_name_no_results(self, vector_store):
        """Test course name resolution with no results."""
        vector_store.course_catalog.query.return_value = {
            "documents": [[]],
            "metadatas": [[]]
        }

        resolved = vector_store._resolve_course_name("NonExistent")

        assert resolved is None

    def test_resolve_course_name_exception(self, vector_store):
        """Test course name resolution exception handling."""
        vector_store.course_catalog.query.side_effect = Exception("Error")

        resolved = vector_store._resolve_course_name("Test")

        assert resolved is None

    def test_build_filter_no_parameters(self, vector_store):
        """Test filter building with no parameters."""
        filter_dict = vector_store._build_filter(None, None)
        assert filter_dict is None

    def test_build_filter_course_only(self, vector_store):
        """Test filter building with course only."""
        filter_dict = vector_store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, vector_store):
        """Test filter building with lesson only."""
        filter_dict = vector_store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}

    def test_build_filter_both_parameters(self, vector_store):
        """Test filter building with both parameters."""
        filter_dict = vector_store._build_filter("Test Course", 1)
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 1}
            ]
        }
        assert filter_dict == expected

    def test_get_existing_course_titles(self, vector_store):
        """Test getting existing course titles."""
        vector_store.course_catalog.get.return_value = {
            "ids": ["Course 1", "Course 2", "Course 3"]
        }

        titles = vector_store.get_existing_course_titles()

        assert titles == ["Course 1", "Course 2", "Course 3"]

    def test_get_existing_course_titles_empty(self, vector_store):
        """Test getting course titles when none exist."""
        vector_store.course_catalog.get.return_value = {"ids": []}

        titles = vector_store.get_existing_course_titles()

        assert titles == []

    def test_get_existing_course_titles_exception(self, vector_store):
        """Test getting course titles with exception."""
        vector_store.course_catalog.get.side_effect = Exception("Error")

        titles = vector_store.get_existing_course_titles()

        assert titles == []

    def test_get_course_count(self, vector_store):
        """Test getting course count."""
        vector_store.course_catalog.get.return_value = {
            "ids": ["Course 1", "Course 2"]
        }

        count = vector_store.get_course_count()

        assert count == 2

    def test_get_course_count_exception(self, vector_store):
        """Test getting course count with exception."""
        vector_store.course_catalog.get.side_effect = Exception("Error")

        count = vector_store.get_course_count()

        assert count == 0

    def test_get_all_courses_metadata(self, vector_store):
        """Test getting all courses metadata."""
        lessons_data = [
            {"lesson_number": 1, "lesson_title": "Intro", "lesson_link": "http://link1"},
            {"lesson_number": 2, "lesson_title": "Advanced", "lesson_link": "http://link2"}
        ]

        vector_store.course_catalog.get.return_value = {
            "metadatas": [{
                "title": "Test Course",
                "instructor": "Test Instructor",
                "lessons_json": json.dumps(lessons_data),
                "lesson_count": 2
            }]
        }

        metadata = vector_store.get_all_courses_metadata()

        assert len(metadata) == 1
        course_meta = metadata[0]
        assert course_meta["title"] == "Test Course"
        assert course_meta["instructor"] == "Test Instructor"
        assert course_meta["lessons"] == lessons_data
        assert "lessons_json" not in course_meta  # Should be removed

    def test_get_course_link(self, vector_store):
        """Test getting course link."""
        vector_store.course_catalog.get.return_value = {
            "metadatas": [{"course_link": "https://example.com/course"}]
        }

        link = vector_store.get_course_link("Test Course")

        assert link == "https://example.com/course"

    def test_get_course_link_not_found(self, vector_store):
        """Test getting course link when not found."""
        vector_store.course_catalog.get.return_value = {"metadatas": []}

        link = vector_store.get_course_link("NonExistent")

        assert link is None

    def test_get_lesson_link(self, vector_store):
        """Test getting lesson link."""
        lessons_data = [
            {"lesson_number": 1, "lesson_link": "https://example.com/lesson/1"},
            {"lesson_number": 2, "lesson_link": "https://example.com/lesson/2"}
        ]

        vector_store.course_catalog.get.return_value = {
            "metadatas": [{
                "lessons_json": json.dumps(lessons_data)
            }]
        }

        link = vector_store.get_lesson_link("Test Course", 1)

        assert link == "https://example.com/lesson/1"

    def test_get_lesson_link_not_found(self, vector_store):
        """Test getting lesson link when lesson not found."""
        lessons_data = [
            {"lesson_number": 1, "lesson_link": "https://example.com/lesson/1"}
        ]

        vector_store.course_catalog.get.return_value = {
            "metadatas": [{
                "lessons_json": json.dumps(lessons_data)
            }]
        }

        link = vector_store.get_lesson_link("Test Course", 99)

        assert link is None

    def test_clear_all_data(self, vector_store, mock_chromadb):
        """Test clearing all data."""
        vector_store.clear_all_data()

        # Should delete both collections
        vector_store.client.delete_collection.assert_any_call("course_catalog")
        vector_store.client.delete_collection.assert_any_call("course_content")

        # Should recreate collections
        assert vector_store.client.get_or_create_collection.call_count >= 2

    def test_clear_all_data_exception(self, vector_store):
        """Test clearing all data with exception."""
        vector_store.client.delete_collection.side_effect = Exception("Delete error")

        # Should not raise exception
        vector_store.clear_all_data()

    def test_search_with_custom_limit(self, vector_store):
        """Test search with custom limit."""
        vector_store.course_content.query.return_value = {
            "documents": [["result"]],
            "metadatas": [[{}]],
            "distances": [[0.5]]
        }

        vector_store.search("test query", limit=10)

        call_args = vector_store.course_content.query.call_args
        assert call_args[1]["n_results"] == 10

    def test_search_with_default_limit(self, vector_store):
        """Test search uses default max_results when no limit specified."""
        vector_store.course_content.query.return_value = {
            "documents": [["result"]],
            "metadatas": [[{}]],
            "distances": [[0.5]]
        }

        vector_store.search("test query")

        call_args = vector_store.course_content.query.call_args
        assert call_args[1]["n_results"] == vector_store.max_results