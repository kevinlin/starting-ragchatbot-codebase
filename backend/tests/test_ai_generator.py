"""
Unit tests for AIGenerator class.
Tests AI response generation, tool execution, and Anthropic API integration.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test the AIGenerator class."""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create a mock Anthropic client."""
        client = Mock()
        return client

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create an AIGenerator for testing."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_anthropic.return_value = mock_anthropic_client
            return AIGenerator(api_key="test_key", model="claude-3-sonnet-20240229")

    def test_initialization(self):
        """Test AIGenerator initialization."""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            ai_gen = AIGenerator(api_key="test_api_key", model="test_model")

            mock_anthropic.assert_called_once_with(api_key="test_api_key")
            assert ai_gen.model == "test_model"
            assert ai_gen.base_params["model"] == "test_model"
            assert ai_gen.base_params["temperature"] == 0
            assert ai_gen.base_params["max_tokens"] == 800

    def test_system_prompt_constant(self):
        """Test that SYSTEM_PROMPT is properly defined."""
        assert AIGenerator.SYSTEM_PROMPT is not None
        assert len(AIGenerator.SYSTEM_PROMPT) > 0
        assert "tools" in AIGenerator.SYSTEM_PROMPT.lower()
        assert "course content search" in AIGenerator.SYSTEM_PROMPT.lower()
        assert "course outline" in AIGenerator.SYSTEM_PROMPT.lower()

    def test_generate_response_basic(self, ai_generator, mock_anthropic_client):
        """Test basic response generation without tools."""
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a basic response."
        mock_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.return_value = mock_response

        result = ai_generator.generate_response("What is AI?")

        # Verify API call
        mock_anthropic_client.messages.create.assert_called_once()
        call_args = mock_anthropic_client.messages.create.call_args

        # Check parameters
        assert call_args[1]["model"] == "claude-3-sonnet-20240229"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert call_args[1]["messages"] == [{"role": "user", "content": "What is AI?"}]
        assert call_args[1]["system"] == AIGenerator.SYSTEM_PROMPT

        assert result == "This is a basic response."

    def test_generate_response_with_history(self, ai_generator, mock_anthropic_client):
        """Test response generation with conversation history."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with history."
        mock_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.return_value = mock_response

        history = "User: Previous question\nAssistant: Previous answer"
        result = ai_generator.generate_response("Follow-up question", conversation_history=history)

        call_args = mock_anthropic_client.messages.create.call_args
        system_content = call_args[1]["system"]

        # Should include history in system content
        assert "Previous conversation:" in system_content
        assert history in system_content

    def test_generate_response_with_tools(self, ai_generator, mock_anthropic_client):
        """Test response generation with tools available."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with tools available."
        mock_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.return_value = mock_response

        tools = [{
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {"type": "object"}
        }]

        result = ai_generator.generate_response("Test query", tools=tools)

        call_args = mock_anthropic_client.messages.create.call_args

        # Should include tools in API call
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}

    def test_generate_response_with_tool_use(self, ai_generator, mock_anthropic_client):
        """Test response generation that triggers tool use."""
        # Mock initial response with tool use
        initial_response = Mock()
        initial_response.stop_reason = "tool_use"

        # Mock tool use content block
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.input = {"query": "AI concepts"}
        tool_block.id = "tool_123"

        # Mock text content block
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "I'll search for information."

        initial_response.content = [text_block, tool_block]

        # Mock final response after tool execution
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Based on the search results, AI is..."

        mock_anthropic_client.messages.create.side_effect = [initial_response, final_response]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about AI"

        tools = [{"name": "search_course_content"}]

        result = ai_generator.generate_response(
            "What is AI?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should have made two API calls
        assert mock_anthropic_client.messages.create.call_count == 2

        # Should have executed the tool
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="AI concepts"
        )

        assert result == "Based on the search results, AI is..."

    def test_handle_tool_execution_single_tool(self, ai_generator, mock_anthropic_client):
        """Test handling single tool execution."""
        # Mock initial response with tool use
        initial_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "test_tool"
        tool_block.input = {"param": "value"}
        tool_block.id = "tool_456"
        initial_response.content = [tool_block]

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Tool execution complete."
        mock_anthropic_client.messages.create.return_value = final_response

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        base_params = {
            "model": "test_model",
            "messages": [{"role": "user", "content": "Original query"}],
            "system": "Test system prompt"
        }

        result = ai_generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with("test_tool", param="value")

        # Verify final API call structure
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]

        # Should have 3 messages: original, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0] == {"role": "user", "content": "Original query"}
        assert messages[1] == {"role": "assistant", "content": initial_response.content}
        assert messages[2]["role"] == "user"

        # Tool results should be properly formatted
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_456"
        assert tool_results[0]["content"] == "Tool result"

        assert result == "Tool execution complete."

    def test_handle_tool_execution_multiple_tools(self, ai_generator, mock_anthropic_client):
        """Test handling multiple tool executions in one response."""
        # Mock initial response with multiple tool uses
        initial_response = Mock()

        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "tool_1"
        tool_block1.input = {"param1": "value1"}
        tool_block1.id = "tool_1_id"

        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "tool_2"
        tool_block2.input = {"param2": "value2"}
        tool_block2.id = "tool_2_id"

        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Using multiple tools"

        initial_response.content = [text_block, tool_block1, tool_block2]

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Multiple tools executed."
        mock_anthropic_client.messages.create.return_value = final_response

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        base_params = {
            "model": "test_model",
            "messages": [{"role": "user", "content": "Multi-tool query"}],
            "system": "Test system prompt"
        }

        result = ai_generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("tool_1", param1="value1")
        mock_tool_manager.execute_tool.assert_any_call("tool_2", param2="value2")

        # Verify tool results structure
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]
        tool_results = messages[2]["content"]

        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_1_id"
        assert tool_results[0]["content"] == "Result 1"
        assert tool_results[1]["tool_use_id"] == "tool_2_id"
        assert tool_results[1]["content"] == "Result 2"

    def test_handle_tool_execution_no_tool_blocks(self, ai_generator, mock_anthropic_client):
        """Test handling when no tool use blocks are present."""
        # Mock response with only text (shouldn't happen but test defensive coding)
        initial_response = Mock()
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "No tools here"
        initial_response.content = [text_block]

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "No tools processed."
        mock_anthropic_client.messages.create.return_value = final_response

        mock_tool_manager = Mock()

        base_params = {
            "model": "test_model",
            "messages": [{"role": "user", "content": "Query"}],
            "system": "Test system"
        }

        result = ai_generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)

        # No tools should be executed
        mock_tool_manager.execute_tool.assert_not_called()

        # Should still make final API call with empty tool results
        call_args = mock_anthropic_client.messages.create.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2  # original + assistant, no tool results

    def test_generate_response_without_conversation_history(self, ai_generator, mock_anthropic_client):
        """Test that system prompt is used correctly without history."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response without history."
        mock_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response("Test query", conversation_history=None)

        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["system"] == AIGenerator.SYSTEM_PROMPT

    def test_generate_response_empty_conversation_history(self, ai_generator, mock_anthropic_client):
        """Test handling of empty conversation history."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with empty history."
        mock_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.return_value = mock_response

        ai_generator.generate_response("Test query", conversation_history="")

        call_args = mock_anthropic_client.messages.create.call_args
        # Empty string should be treated as no history
        assert call_args[1]["system"] == AIGenerator.SYSTEM_PROMPT

    def test_base_params_immutability(self, ai_generator, mock_anthropic_client):
        """Test that base_params aren't mutated during calls."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response."
        mock_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.return_value = mock_response

        original_base_params = ai_generator.base_params.copy()

        ai_generator.generate_response("Test query")

        # base_params should remain unchanged
        assert ai_generator.base_params == original_base_params

    def test_tool_execution_with_no_tool_manager(self, ai_generator, mock_anthropic_client):
        """Test that tool use without tool_manager returns the initial response."""
        # Mock response with tool use
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "I would use a tool here."
        mock_response.stop_reason = "tool_use"

        mock_anthropic_client.messages.create.return_value = mock_response

        tools = [{"name": "test_tool"}]

        result = ai_generator.generate_response("Test query", tools=tools, tool_manager=None)

        # Should return the initial response text since no tool_manager provided
        assert result == "I would use a tool here."

    def test_error_handling_in_tool_execution(self, ai_generator, mock_anthropic_client):
        """Test error handling during tool execution."""
        # Mock initial response
        initial_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "failing_tool"
        tool_block.input = {"param": "value"}
        tool_block.id = "tool_id"
        initial_response.content = [tool_block]

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Handled tool error."
        mock_anthropic_client.messages.create.return_value = final_response

        # Mock tool manager that raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        base_params = {
            "model": "test_model",
            "messages": [{"role": "user", "content": "Query"}],
            "system": "Test system"
        }

        # Should not raise exception, should handle gracefully
        result = ai_generator._handle_tool_execution(initial_response, base_params, mock_tool_manager)

        # The exception should bubble up since we don't handle it in the method
        # This is the current behavior - if tools fail, the error should be visible

    def test_api_parameter_structure(self, ai_generator, mock_anthropic_client):
        """Test that API parameters are structured correctly."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Parameter test response."
        mock_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.return_value = mock_response

        tools = [{"name": "test_tool", "description": "Test"}]
        history = "Previous: context"

        ai_generator.generate_response(
            query="Test query",
            conversation_history=history,
            tools=tools
        )

        call_args = mock_anthropic_client.messages.create.call_args[1]

        # Verify all expected parameters are present
        assert "model" in call_args
        assert "temperature" in call_args
        assert "max_tokens" in call_args
        assert "messages" in call_args
        assert "system" in call_args
        assert "tools" in call_args
        assert "tool_choice" in call_args

        # Verify message structure
        assert call_args["messages"] == [{"role": "user", "content": "Test query"}]

        # Verify system includes history
        assert "Previous conversation:" in call_args["system"]
        assert history in call_args["system"]

    def test_system_prompt_construction(self, ai_generator):
        """Test system prompt construction with and without history."""
        # Test without history
        result_without = ai_generator._build_system_content(None)
        assert result_without == AIGenerator.SYSTEM_PROMPT

        # Test with history
        history = "User: Hi\nAssistant: Hello"
        result_with = ai_generator._build_system_content(history)
        assert AIGenerator.SYSTEM_PROMPT in result_with
        assert "Previous conversation:" in result_with
        assert history in result_with

    def _build_system_content(self, conversation_history):
        """Helper method to test system content building (add to ai_generator.py if needed)."""
        return (
            f"{AIGenerator.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else AIGenerator.SYSTEM_PROMPT
        )

    def test_sequential_tool_calling_two_rounds(self, ai_generator, mock_anthropic_client):
        """Test successful 2-round sequential tool calling."""
        # Mock first round response - Claude uses get_course_outline
        first_response = Mock()
        first_response.stop_reason = "tool_use"

        first_tool_block = Mock()
        first_tool_block.type = "tool_use"
        first_tool_block.name = "get_course_outline"
        first_tool_block.input = {"course_name": "AI Fundamentals"}
        first_tool_block.id = "tool_1"

        first_response.content = [first_tool_block]

        # Mock second round response - Claude uses search_course_content
        second_response = Mock()
        second_response.stop_reason = "tool_use"

        second_tool_block = Mock()
        second_tool_block.type = "tool_use"
        second_tool_block.name = "search_course_content"
        second_tool_block.input = {"query": "machine learning basics"}
        second_tool_block.id = "tool_2"

        second_response.content = [second_tool_block]

        # Mock final response - Claude provides final answer
        final_response = Mock()
        final_response.stop_reason = "end_turn"
        final_response.content = [Mock()]
        final_response.content[0].text = "Based on the course outline and search results, here's the answer..."

        mock_anthropic_client.messages.create.side_effect = [
            first_response, second_response, final_response
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline: 1. Introduction 2. ML Basics",
            "Machine learning content from multiple courses"
        ]

        tools = [{"name": "get_course_outline"}, {"name": "search_course_content"}]

        result = ai_generator.generate_response(
            "Find courses that cover the same topics as lesson 2 of AI Fundamentals",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should have made 3 API calls (2 tool rounds + 1 final)
        assert mock_anthropic_client.messages.create.call_count == 3

        # Should have executed both tools
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="AI Fundamentals")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="machine learning basics")

        assert result == "Based on the course outline and search results, here's the answer..."

    def test_sequential_tool_calling_early_termination(self, ai_generator, mock_anthropic_client):
        """Test sequential tool calling with early termination (no tool use in first round)."""
        # Mock response without tool use - Claude decides no tools needed
        response = Mock()
        response.stop_reason = "end_turn"
        response.content = [Mock()]
        response.content[0].text = "I can answer this without using tools."

        mock_anthropic_client.messages.create.return_value = response

        mock_tool_manager = Mock()
        tools = [{"name": "search_course_content"}]

        result = ai_generator.generate_response(
            "What is machine learning?",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should have made only 1 API call
        assert mock_anthropic_client.messages.create.call_count == 1

        # No tools should have been executed
        mock_tool_manager.execute_tool.assert_not_called()

        assert result == "I can answer this without using tools."

    def test_sequential_tool_calling_max_rounds_reached(self, ai_generator, mock_anthropic_client):
        """Test that sequential calling stops after MAX_TOOL_ROUNDS."""
        # Mock responses for both rounds with tool use
        round_1_response = Mock()
        round_1_response.stop_reason = "tool_use"
        tool_block_1 = Mock()
        tool_block_1.type = "tool_use"
        tool_block_1.name = "search_course_content"
        tool_block_1.input = {"query": "python basics"}
        tool_block_1.id = "tool_1"
        round_1_response.content = [tool_block_1]

        round_2_response = Mock()
        round_2_response.stop_reason = "tool_use"
        tool_block_2 = Mock()
        tool_block_2.type = "tool_use"
        tool_block_2.name = "get_course_outline"
        tool_block_2.input = {"course_name": "Python Essentials"}
        tool_block_2.id = "tool_2"
        round_2_response.content = [tool_block_2]

        # Mock final response after max rounds
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final answer after max rounds reached."

        mock_anthropic_client.messages.create.side_effect = [
            round_1_response, round_2_response, final_response
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        tools = [{"name": "search_course_content"}, {"name": "get_course_outline"}]

        result = ai_generator.generate_response(
            "Complex query requiring multiple searches",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should have made 3 calls: 2 rounds + final without tools
        assert mock_anthropic_client.messages.create.call_count == 3

        # Should have executed tools from both rounds
        assert mock_tool_manager.execute_tool.call_count == 2

        assert result == "Final answer after max rounds reached."

    def test_sequential_tool_calling_tool_failure_in_first_round(self, ai_generator, mock_anthropic_client):
        """Test handling of tool failure in first round of sequential calling."""
        # Mock first round with tool use
        first_response = Mock()
        first_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "failing_tool"
        tool_block.input = {"param": "value"}
        tool_block.id = "tool_fail"
        first_response.content = [tool_block]

        # Mock final response after tool failure
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Handled tool failure gracefully."

        mock_anthropic_client.messages.create.side_effect = [first_response, final_response]

        # Mock tool manager that fails
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        tools = [{"name": "failing_tool"}]

        result = ai_generator.generate_response(
            "Query that triggers tool failure",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should have made 2 calls: 1 with tool failure + 1 final
        assert mock_anthropic_client.messages.create.call_count == 2

        # Tool should have been attempted once
        mock_tool_manager.execute_tool.assert_called_once()

        assert result == "Handled tool failure gracefully."

    def test_sequential_tool_calling_partial_tool_failure(self, ai_generator, mock_anthropic_client):
        """Test handling partial tool failure (some tools succeed, others fail)."""
        # Mock response with multiple tool calls
        first_response = Mock()
        first_response.stop_reason = "tool_use"

        success_tool = Mock()
        success_tool.type = "tool_use"
        success_tool.name = "working_tool"
        success_tool.input = {"query": "test"}
        success_tool.id = "tool_success"

        fail_tool = Mock()
        fail_tool.type = "tool_use"
        fail_tool.name = "failing_tool"
        fail_tool.input = {"param": "test"}
        fail_tool.id = "tool_fail"

        first_response.content = [success_tool, fail_tool]

        # Mock second round response
        second_response = Mock()
        second_response.stop_reason = "end_turn"
        second_response.content = [Mock()]
        second_response.content[0].text = "Continued despite partial tool failure."

        mock_anthropic_client.messages.create.side_effect = [first_response, second_response]

        # Mock tool manager with mixed success/failure
        mock_tool_manager = Mock()
        def mock_execute_tool(name, **kwargs):
            if name == "working_tool":
                return "Success result"
            else:
                raise Exception("This tool failed")

        mock_tool_manager.execute_tool.side_effect = mock_execute_tool

        tools = [{"name": "working_tool"}, {"name": "failing_tool"}]

        result = ai_generator.generate_response(
            "Query with partial tool failure",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should have made 2 calls (continued to second round despite partial failure)
        assert mock_anthropic_client.messages.create.call_count == 2

        # Both tools should have been attempted
        assert mock_tool_manager.execute_tool.call_count == 2

        assert result == "Continued despite partial tool failure."

    def test_sequential_tool_calling_api_failure_recovery(self, ai_generator, mock_anthropic_client):
        """Test recovery from API failures during sequential rounds."""
        # Mock successful first round
        first_response = Mock()
        first_response.stop_reason = "end_turn"
        first_response.content = [Mock()]
        first_response.content[0].text = "First round successful response."

        # Mock API failure on second call
        mock_anthropic_client.messages.create.side_effect = [
            first_response,
            Exception("API call failed")
        ]

        mock_tool_manager = Mock()
        tools = [{"name": "test_tool"}]

        result = ai_generator.generate_response(
            "Query that causes API failure",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        # Should return the last successful response
        assert result == "First round successful response."

    def test_max_tool_rounds_constant(self, ai_generator):
        """Test that MAX_TOOL_ROUNDS constant is properly defined."""
        assert hasattr(AIGenerator, 'MAX_TOOL_ROUNDS')
        assert AIGenerator.MAX_TOOL_ROUNDS == 2
        assert ai_generator.MAX_TOOL_ROUNDS == 2

    def test_backward_compatibility_single_round_with_tools(self, ai_generator, mock_anthropic_client):
        """Test that single-round behavior is preserved when tool_manager is None."""
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Single round response."
        mock_response.stop_reason = "end_turn"

        mock_anthropic_client.messages.create.return_value = mock_response

        tools = [{"name": "test_tool"}]

        # Call without tool_manager should use fallback behavior
        result = ai_generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=None
        )

        # Should make only one API call
        assert mock_anthropic_client.messages.create.call_count == 1

        # Should include tools in the call
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tools" in call_args[1]

        assert result == "Single round response."