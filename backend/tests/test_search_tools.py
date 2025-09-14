"""
Unit tests for SearchTools classes.
Tests CourseSearchTool and ToolManager functionality.
"""
import pytest
from unittest.mock import Mock, patch

from search_tools import Tool, CourseSearchTool, ToolManager
from vector_store import VectorStore, SearchResults


class MockTool(Tool):
    """Mock tool implementation for testing."""

    def __init__(self, name="mock_tool"):
        self.name = name
        self.executed = False
        self.execution_args = None

    def get_tool_definition(self):
        return {
            "name": self.name,
            "description": "Mock tool for testing",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Test parameter"}
                },
                "required": ["param"]
            }
        }

    def execute(self, **kwargs):
        self.executed = True
        self.execution_args = kwargs
        return f"Executed {self.name} with {kwargs}"


class TestTool:
    """Test the Tool abstract base class."""

    def test_tool_interface(self):
        """Test that Tool is properly abstract."""
        # Should not be able to instantiate Tool directly
        with pytest.raises(TypeError):
            Tool()

    def test_mock_tool_implementation(self):
        """Test our mock tool implementation."""
        tool = MockTool("test_tool")

        definition = tool.get_tool_definition()
        assert definition["name"] == "test_tool"
        assert "description" in definition

        result = tool.execute(param="test_value")
        assert tool.executed is True
        assert tool.execution_args == {"param": "test_value"}
        assert "test_tool" in result


class TestCourseSearchTool:
    """Test the CourseSearchTool class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing."""
        store = Mock(spec=VectorStore)
        return store

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create a CourseSearchTool for testing."""
        return CourseSearchTool(mock_vector_store)

    def test_initialization(self, mock_vector_store):
        """Test CourseSearchTool initialization."""
        tool = CourseSearchTool(mock_vector_store)

        assert tool.store == mock_vector_store
        assert tool.last_sources == []

    def test_get_tool_definition(self, search_tool):
        """Test getting tool definition."""
        definition = search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert schema["required"] == ["query"]

    def test_execute_basic_search(self, search_tool):
        """Test basic search execution."""
        # Mock search results
        mock_results = SearchResults(
            documents=["Document content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.5]
        )
        search_tool.store.search.return_value = mock_results
        search_tool.store.get_lesson_link.return_value = "https://example.com/lesson/1"

        result = search_tool.execute(query="test query")

        # Verify search was called
        search_tool.store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )

        # Verify result formatting
        assert "[Test Course - Lesson 1]" in result
        assert "Document content" in result

        # Verify sources are tracked
        assert len(search_tool.last_sources) == 1
        source = search_tool.last_sources[0]
        assert source["text"] == "Test Course - Lesson 1"
        assert source["url"] == "https://example.com/lesson/1"

    def test_execute_with_course_filter(self, search_tool):
        """Test search execution with course filter."""
        mock_results = SearchResults(
            documents=["Course content"],
            metadata=[{"course_title": "Filtered Course"}],
            distances=[0.3]
        )
        search_tool.store.search.return_value = mock_results
        search_tool.store.get_lesson_link.return_value = None

        result = search_tool.execute(query="test query", course_name="Filtered")

        search_tool.store.search.assert_called_once_with(
            query="test query",
            course_name="Filtered",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, search_tool):
        """Test search execution with lesson filter."""
        mock_results = SearchResults(
            documents=["Lesson content"],
            metadata=[{"course_title": "Test Course", "lesson_number": 2}],
            distances=[0.2]
        )
        search_tool.store.search.return_value = mock_results
        search_tool.store.get_lesson_link.return_value = "https://example.com/lesson/2"

        result = search_tool.execute(query="test query", lesson_number=2)

        search_tool.store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=2
        )

    def test_execute_with_all_parameters(self, search_tool):
        """Test search execution with all parameters."""
        mock_results = SearchResults(
            documents=["Specific content"],
            metadata=[{"course_title": "Specific Course", "lesson_number": 3}],
            distances=[0.1]
        )
        search_tool.store.search.return_value = mock_results
        search_tool.store.get_lesson_link.return_value = "https://example.com/lesson/3"

        result = search_tool.execute(
            query="specific query",
            course_name="Specific",
            lesson_number=3
        )

        search_tool.store.search.assert_called_once_with(
            query="specific query",
            course_name="Specific",
            lesson_number=3
        )

    def test_execute_with_error_result(self, search_tool):
        """Test search execution when search returns error."""
        error_results = SearchResults.empty("Search failed")
        error_results.error = "Database connection error"
        search_tool.store.search.return_value = error_results

        result = search_tool.execute(query="test query")

        assert result == "Database connection error"
        assert search_tool.last_sources == []

    def test_execute_with_empty_results(self, search_tool):
        """Test search execution with no results."""
        empty_results = SearchResults([], [], [])
        search_tool.store.search.return_value = empty_results

        result = search_tool.execute(query="test query")

        assert result == "No relevant content found."

    def test_execute_with_empty_results_and_filters(self, search_tool):
        """Test search execution with no results but with filters."""
        empty_results = SearchResults([], [], [])
        search_tool.store.search.return_value = empty_results

        result = search_tool.execute(
            query="test query",
            course_name="NonExistent",
            lesson_number=99
        )

        expected = "No relevant content found in course 'NonExistent' in lesson 99."
        assert result == expected

    def test_format_results_with_links(self, search_tool):
        """Test result formatting with lesson links."""
        results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )

        search_tool.store.get_lesson_link.side_effect = [
            "https://example.com/lesson/1",
            "https://example.com/lesson/2"
        ]

        formatted = search_tool._format_results(results)

        assert "[Course A - Lesson 1]" in formatted
        assert "[Course B - Lesson 2]" in formatted
        assert "Content 1" in formatted
        assert "Content 2" in formatted

        # Check sources
        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["url"] == "https://example.com/lesson/1"
        assert search_tool.last_sources[1]["url"] == "https://example.com/lesson/2"

    def test_format_results_without_links(self, search_tool):
        """Test result formatting without lesson links."""
        results = SearchResults(
            documents=["Content without link"],
            metadata=[{"course_title": "Course C"}],  # No lesson_number
            distances=[0.3]
        )

        search_tool.store.get_lesson_link.return_value = None

        formatted = search_tool._format_results(results)

        assert "[Course C]" in formatted
        assert "Content without link" in formatted

        # Check sources (no link)
        assert len(search_tool.last_sources) == 1
        source = search_tool.last_sources[0]
        assert source["text"] == "Course C"
        assert "url" not in source

    def test_format_results_mixed_links(self, search_tool):
        """Test result formatting with mixed link availability."""
        results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B"}  # No lesson_number
            ],
            distances=[0.1, 0.2]
        )

        def mock_get_lesson_link(course_title, lesson_number):
            if lesson_number == 1:
                return "https://example.com/lesson/1"
            return None

        search_tool.store.get_lesson_link.side_effect = mock_get_lesson_link

        formatted = search_tool._format_results(results)

        # Check sources
        assert len(search_tool.last_sources) == 2

        # First source should have link
        source1 = search_tool.last_sources[0]
        assert source1["text"] == "Course A - Lesson 1"
        assert source1["url"] == "https://example.com/lesson/1"

        # Second source should not have link
        source2 = search_tool.last_sources[1]
        assert source2["text"] == "Course B"
        assert "url" not in source2

    def test_format_results_unknown_course(self, search_tool):
        """Test result formatting with unknown course."""
        results = SearchResults(
            documents=["Mystery content"],
            metadata=[{}],  # Empty metadata
            distances=[0.5]
        )

        formatted = search_tool._format_results(results)

        assert "[unknown]" in formatted
        assert "Mystery content" in formatted

    def test_multiple_executions_reset_sources(self, search_tool):
        """Test that sources are properly replaced on multiple executions."""
        # First execution
        results1 = SearchResults(
            documents=["Content 1"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1]
        )
        search_tool.store.search.return_value = results1
        search_tool.store.get_lesson_link.return_value = "https://example.com/lesson/1"

        search_tool.execute(query="query 1")
        assert len(search_tool.last_sources) == 1

        # Second execution should replace sources
        results2 = SearchResults(
            documents=["Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course 2", "lesson_number": 1},
                {"course_title": "Course 2", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        search_tool.store.search.return_value = results2

        search_tool.execute(query="query 2")
        assert len(search_tool.last_sources) == 2
        assert all("Course 2" in source["text"] for source in search_tool.last_sources)


class TestToolManager:
    """Test the ToolManager class."""

    @pytest.fixture
    def tool_manager(self):
        """Create a ToolManager for testing."""
        return ToolManager()

    def test_initialization(self, tool_manager):
        """Test ToolManager initialization."""
        assert tool_manager.tools == {}

    def test_register_tool(self, tool_manager):
        """Test registering a tool."""
        mock_tool = MockTool("test_tool")

        tool_manager.register_tool(mock_tool)

        assert "test_tool" in tool_manager.tools
        assert tool_manager.tools["test_tool"] == mock_tool

    def test_register_tool_without_name(self, tool_manager):
        """Test registering a tool without name raises error."""
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"description": "No name"}

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            tool_manager.register_tool(mock_tool)

    def test_register_multiple_tools(self, tool_manager):
        """Test registering multiple tools."""
        tool1 = MockTool("tool_1")
        tool2 = MockTool("tool_2")

        tool_manager.register_tool(tool1)
        tool_manager.register_tool(tool2)

        assert len(tool_manager.tools) == 2
        assert "tool_1" in tool_manager.tools
        assert "tool_2" in tool_manager.tools

    def test_get_tool_definitions(self, tool_manager):
        """Test getting all tool definitions."""
        tool1 = MockTool("tool_1")
        tool2 = MockTool("tool_2")

        tool_manager.register_tool(tool1)
        tool_manager.register_tool(tool2)

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 2
        tool_names = [def_["name"] for def_ in definitions]
        assert "tool_1" in tool_names
        assert "tool_2" in tool_names

    def test_get_tool_definitions_empty(self, tool_manager):
        """Test getting tool definitions when no tools registered."""
        definitions = tool_manager.get_tool_definitions()
        assert definitions == []

    def test_execute_tool(self, tool_manager):
        """Test executing a registered tool."""
        mock_tool = MockTool("test_tool")
        tool_manager.register_tool(mock_tool)

        result = tool_manager.execute_tool("test_tool", param="test_value")

        assert mock_tool.executed is True
        assert mock_tool.execution_args == {"param": "test_value"}
        assert "test_tool" in result

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test executing a non-existent tool."""
        result = tool_manager.execute_tool("nonexistent_tool", param="value")

        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources_single_tool(self, tool_manager):
        """Test getting last sources from single tool."""
        mock_vector_store = Mock(spec=VectorStore)
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = [{"text": "Test source", "url": "http://test"}]

        tool_manager.register_tool(search_tool)

        sources = tool_manager.get_last_sources()

        assert sources == [{"text": "Test source", "url": "http://test"}]

    def test_get_last_sources_multiple_tools_one_with_sources(self, tool_manager):
        """Test getting last sources when multiple tools exist."""
        # Tool without sources
        regular_tool = MockTool("regular_tool")

        # Tool with sources
        mock_vector_store = Mock(spec=VectorStore)
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = [{"text": "Search source"}]

        tool_manager.register_tool(regular_tool)
        tool_manager.register_tool(search_tool)

        sources = tool_manager.get_last_sources()

        assert sources == [{"text": "Search source"}]

    def test_get_last_sources_no_sources(self, tool_manager):
        """Test getting last sources when no tools have sources."""
        tool1 = MockTool("tool_1")
        tool2 = MockTool("tool_2")

        tool_manager.register_tool(tool1)
        tool_manager.register_tool(tool2)

        sources = tool_manager.get_last_sources()

        assert sources == []

    def test_reset_sources(self, tool_manager):
        """Test resetting sources from all tools."""
        # Create search tool with sources
        mock_vector_store = Mock(spec=VectorStore)
        search_tool = CourseSearchTool(mock_vector_store)
        search_tool.last_sources = [{"text": "Test source"}]

        # Create regular tool (no sources)
        regular_tool = MockTool("regular_tool")

        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(regular_tool)

        # Verify sources exist
        assert search_tool.last_sources != []

        # Reset sources
        tool_manager.reset_sources()

        # Verify sources are cleared
        assert search_tool.last_sources == []

    def test_reset_sources_no_tools(self, tool_manager):
        """Test resetting sources when no tools exist."""
        # Should not raise an exception
        tool_manager.reset_sources()

    def test_tool_registration_overwrite(self, tool_manager):
        """Test that registering a tool with same name overwrites."""
        tool1 = MockTool("same_name")
        tool2 = MockTool("same_name")

        tool_manager.register_tool(tool1)
        tool_manager.register_tool(tool2)

        assert len(tool_manager.tools) == 1
        assert tool_manager.tools["same_name"] == tool2

    def test_integration_with_course_search_tool(self, tool_manager):
        """Test integration between ToolManager and CourseSearchTool."""
        # Create a real CourseSearchTool with mock vector store
        mock_vector_store = Mock(spec=VectorStore)
        mock_results = SearchResults(
            documents=["Integration test content"],
            metadata=[{"course_title": "Integration Course", "lesson_number": 1}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = mock_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/integration"

        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Execute through ToolManager
        result = tool_manager.execute_tool(
            "search_course_content",
            query="integration test",
            course_name="Integration"
        )

        # Verify execution
        assert "Integration test content" in result
        assert "[Integration Course - Lesson 1]" in result

        # Verify sources are available
        sources = tool_manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["text"] == "Integration Course - Lesson 1"
        assert sources[0]["url"] == "https://example.com/integration"