import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Maximum number of sequential tool calling rounds per query
    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Available Tools:
1. **Course Content Search**: For questions about specific course content, lessons, or detailed educational materials
2. **Course Outline**: For questions about course structure, lesson lists, course overviews, or what topics/lessons a course covers

Tool Usage Guidelines:
- **Course outline questions**: Use get_course_outline for questions like "What does course X cover?", "Show me the lessons for course Y", "What's the structure of course Z?"
- **Content search questions**: Use search_course_content for questions about specific topics, concepts, or detailed educational materials within courses
- **Sequential tool usage**: You may use up to 2 rounds of tool calls per query to handle complex questions requiring multiple searches
- **Follow-up searches**: After reviewing initial tool results, you may make additional tool calls to refine or expand your search based on the information found
- **Complex queries**: Use sequential searches for comparisons, multi-part questions, or when information from different courses/lessons is needed
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Get course outline first, then answer
- **Course-specific content questions**: Search content first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the course outline"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports sequential tool calling up to MAX_TOOL_ROUNDS when tools and tool_manager are provided.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare initial messages
        initial_messages = [{"role": "user", "content": query}]

        # If tools and tool_manager are available, use sequential rounds approach
        if tools and tool_manager:
            return self._execute_sequential_rounds(
                initial_messages=initial_messages,
                system_content=system_content,
                tools=tools,
                tool_manager=tool_manager
            )

        # Fallback to original single-round behavior for backward compatibility
        api_params = {
            **self.base_params,
            "messages": initial_messages,
            "system": system_content
        }

        # Add tools if available (but no tool_manager)
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Handle tool execution if needed (single round only)
        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_tool_execution(response, api_params, tool_manager)

        # Return direct response
        return response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, 
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": f"Tool execution failed: {str(e)}"
                    })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _execute_sequential_rounds(self, initial_messages: List[Dict[str, Any]],
                                 system_content: str, tools: Optional[List],
                                 tool_manager) -> str:
        """
        Execute up to MAX_TOOL_ROUNDS of tool calls, allowing Claude to reason about
        previous results and make follow-up tool calls.

        Args:
            initial_messages: Starting message history
            system_content: System prompt content
            tools: Available tools for Claude to use
            tool_manager: Manager to execute tools

        Returns:
            Final response text after all rounds complete
        """
        messages = initial_messages.copy()
        last_successful_response = None

        for _ in range(self.MAX_TOOL_ROUNDS):
            try:
                # Prepare API call with tools available
                api_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": system_content
                }

                if tools:
                    api_params["tools"] = tools
                    api_params["tool_choice"] = {"type": "auto"}

                # Make API call for this round
                response = self.client.messages.create(**api_params)

                # If no tool use, Claude is done - return response
                if response.stop_reason != "tool_use":
                    return self._extract_text_from_response(response)

                # Store this as last successful response in case of later failures
                last_successful_response = self._extract_text_from_response(response)

                # Claude wants to use tools - execute them
                if not tool_manager:
                    # No tool manager available, return what we have
                    return self._extract_text_from_response(response)

                # Add Claude's response (with tool calls) to message history
                messages.append({"role": "assistant", "content": response.content})

                # Execute all tool calls in this response
                tool_results = []
                tools_executed_successfully = False
                partial_tool_failure = False

                for content_block in response.content:
                    if content_block.type == "tool_use":
                        try:
                            tool_result = tool_manager.execute_tool(
                                content_block.name,
                                **content_block.input
                            )

                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result
                            })
                            tools_executed_successfully = True

                        except Exception as e:
                            # Handle tool execution errors gracefully
                            error_message = f"Tool execution failed: {str(e)}"
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": error_message
                            })
                            partial_tool_failure = True

                # Add tool results to message history if any were generated
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})

                # If no tools executed successfully, stop rounds early
                if not tools_executed_successfully:
                    break

                # If some tools failed but others succeeded, continue but log the issue
                if partial_tool_failure and tools_executed_successfully:
                    # Continue with partial results - Claude may be able to work with what succeeded
                    continue

            except Exception as e:
                # Handle API call failures gracefully
                if last_successful_response:
                    return last_successful_response
                # If no previous successful response, fall back to basic error message
                return f"Unable to process request due to system error: {str(e)}"

        # After max rounds or early termination, make final call without tools
        try:
            final_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            final_response = self.client.messages.create(**final_params)
            return self._extract_text_from_response(final_response)

        except Exception as e:
            # If final API call fails, return last successful response or error
            if last_successful_response:
                return last_successful_response
            return f"Unable to generate final response: {str(e)}"

    def _extract_text_from_response(self, response) -> str:
        """
        Safely extract text content from an API response, handling mixed content types.

        Args:
            response: Anthropic API response object

        Returns:
            Extracted text content or empty string if no text found
        """
        for content_block in response.content:
            if hasattr(content_block, 'text') and content_block.text:
                return content_block.text
        return ""

    def _build_system_content(self, conversation_history: Optional[str] = None) -> str:
        """
        Build system content with optional conversation history.
        
        Args:
            conversation_history: Optional previous conversation context
            
        Returns:
            System prompt content
        """
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )