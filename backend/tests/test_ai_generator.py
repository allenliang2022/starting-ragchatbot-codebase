import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class MockOpenAIResponse:
    """Mock OpenAI response object"""
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.choices = [Mock()]
        self.choices[0].message = Mock()
        self.choices[0].message.content = content
        self.choices[0].message.tool_calls = tool_calls
        self.choices[0].finish_reason = finish_reason


class MockToolCall:
    """Mock tool call object"""
    def __init__(self, id, function_name, arguments):
        self.id = id
        self.function = Mock()
        self.function.name = function_name
        self.function.arguments = arguments


class TestAIGenerator(unittest.TestCase):
    """Test AIGenerator functionality, especially tool integration"""

    def setUp(self):
        """Set up test fixtures"""
        self.ai_generator = AIGenerator(
            api_key="test_key",
            model="gpt-5",
            base_url="http://localhost:4141/v1"
        )

    @patch('ai_generator.OpenAI')
    def test_generate_response_without_tools(self, mock_openai_class):
        """Test basic response generation without tools"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock response
        mock_response = MockOpenAIResponse(content="Test response")
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create new AI generator to use mocked client
        ai_gen = AIGenerator("test_key", "gpt-5", "http://localhost:4141/v1")
        
        # Generate response
        result = ai_gen.generate_response("Test query")
        
        # Verify response
        self.assertEqual(result, "Test response")
        
        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "gpt-5")
        self.assertEqual(len(call_args["messages"]), 2)  # system + user

    @patch('ai_generator.OpenAI')
    def test_generate_response_with_tools_no_tool_calls(self, mock_openai_class):
        """Test response generation with tools available but not used"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock response without tool calls
        mock_response = MockOpenAIResponse(content="Direct response")
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create new AI generator
        ai_gen = AIGenerator("test_key", "gpt-5", "http://localhost:4141/v1")
        
        # Mock tools
        tools = [{
            "name": "search_course_content",
            "description": "Search for course content",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }]
        
        # Generate response
        result = ai_gen.generate_response("Test query", tools=tools)
        
        # Verify response
        self.assertEqual(result, "Direct response")

    @patch('ai_generator.OpenAI')
    def test_generate_response_single_round_backward_compatibility(self, mock_openai_class):
        """Test backward compatibility - single round with max_rounds=1"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Create tool call with JSON arguments
        tool_call = MockToolCall(
            id="call_123",
            function_name="search_course_content",
            arguments='{"query": "Python basics"}'
        )
        
        # Mock initial response with tool calls
        initial_response = MockOpenAIResponse(
            content=None,
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        initial_response.choices[0].message = Mock()
        initial_response.choices[0].message.tool_calls = [tool_call]
        
        # Mock continuation decision response (no more tools)
        decision_response = MockOpenAIResponse(content="Final answer based on tool result")
        
        # Mock final response
        final_response = MockOpenAIResponse(content="Final comprehensive answer")
        
        # Configure mock to return different responses
        mock_client.chat.completions.create.side_effect = [initial_response, decision_response, final_response]
        mock_openai_class.return_value = mock_client
        
        # Create new AI generator
        ai_gen = AIGenerator("test_key", "gpt-5", "http://localhost:4141/v1")
        
        # Mock tools
        tools = [{
            "name": "search_course_content",
            "description": "Search for course content",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }]
        
        # Test with max_rounds=1 for backward compatibility
        result = ai_gen.generate_response("Test query", tools=tools, tool_manager=mock_tool_manager, max_rounds=1)
        
        # Verify it completed successfully
        self.assertEqual(result, "Final answer based on tool result")
        
        # Verify tool was called with correct arguments
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )

    @patch('ai_generator.OpenAI')
    def test_generate_response_sequential_two_rounds(self, mock_openai_class):
        """Test sequential tool calling with two rounds"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["First tool result", "Second tool result"]
        
        # Create tool calls for each round
        first_tool_call = MockToolCall(
            id="call_1",
            function_name="get_course_outline",
            arguments='{"course_name": "Python Course"}'
        )
        
        second_tool_call = MockToolCall(
            id="call_2",
            function_name="search_course_content",
            arguments='{"query": "lesson 4 Python fundamentals"}'
        )
        
        # Mock responses for sequential rounds
        first_response = MockOpenAIResponse(
            content=None,
            tool_calls=[first_tool_call],
            finish_reason="tool_calls"
        )
        first_response.choices[0].message = Mock()
        first_response.choices[0].message.tool_calls = [first_tool_call]
        
        # Mock decision to continue with second round (returns tool calls)
        decision_continue_response = MockOpenAIResponse(
            content=None,
            tool_calls=[second_tool_call],
            finish_reason="tool_calls"
        )
        decision_continue_response.choices[0].message = Mock()
        decision_continue_response.choices[0].message.tool_calls = [second_tool_call]
        
        # Mock second round response
        second_response = MockOpenAIResponse(
            content=None,
            tool_calls=[second_tool_call],
            finish_reason="tool_calls"
        )
        second_response.choices[0].message = Mock()
        second_response.choices[0].message.tool_calls = [second_tool_call]
        
        # Mock final decision to complete (no more tool calls)
        final_decision_response = MockOpenAIResponse(content="Based on both tool results, here's the complete answer")
        
        # Mock final response for when _get_final_response is called
        final_response = MockOpenAIResponse(content="Final comprehensive answer")
        
        # Configure mock responses in sequence
        mock_client.chat.completions.create.side_effect = [
            first_response,              # Round 1: First tool call
            decision_continue_response,  # Round 1: Decision to continue 
            second_response,             # Round 2: Second tool call
            final_decision_response,     # Round 2: Decision to complete
            final_response               # Final response synthesis
        ]
        mock_openai_class.return_value = mock_client
        
        # Create new AI generator
        ai_gen = AIGenerator("test_key", "gpt-5", "http://localhost:4141/v1")
        
        # Mock tools
        tools = [{
            "name": "get_course_outline",
            "description": "Get course outline",
            "input_schema": {"type": "object", "properties": {"course_name": {"type": "string"}}}
        }, {
            "name": "search_course_content",
            "description": "Search for course content",
            "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
        }]
        
        # Test sequential tool calling (default max_rounds=2)
        result = ai_gen.generate_response(
            "Search for a course that discusses the same topic as lesson 4 of Python Course",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        # Verify it completed successfully
        self.assertEqual(result, "Based on both tool results, here's the complete answer")
        
        # Verify both tools were called
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)
        
        # Verify first tool call
        first_call = mock_tool_manager.execute_tool.call_args_list[0]
        self.assertEqual(first_call[0][0], "get_course_outline")
        self.assertEqual(first_call[1]["course_name"], "Python Course")
        
        # Verify second tool call
        second_call = mock_tool_manager.execute_tool.call_args_list[1]
        self.assertEqual(second_call[0][0], "search_course_content")
        self.assertEqual(second_call[1]["query"], "lesson 4 Python fundamentals")

    def test_convert_tools_to_openai_format(self):
        """Test tool format conversion"""
        anthropic_tools = [{
            "name": "search_tool",
            "description": "Search for content",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }]
        
        openai_tools = self.ai_generator._convert_tools_to_openai_format(anthropic_tools)
        
        # Verify conversion
        self.assertEqual(len(openai_tools), 1)
        tool = openai_tools[0]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["function"]["name"], "search_tool")
        self.assertEqual(tool["function"]["description"], "Search for content")
        self.assertEqual(tool["function"]["parameters"]["type"], "object")

    @patch('ai_generator.OpenAI')
    def test_sequential_early_termination_max_rounds(self, mock_openai_class):
        """Test that sequential calling terminates at max_rounds"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        # Create tool call
        tool_call = MockToolCall(
            id="call_123",
            function_name="search_course_content",
            arguments='{"query": "test"}'
        )
        
        # Mock response that always wants to use tools
        tool_response = MockOpenAIResponse(
            content=None,
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        tool_response.choices[0].message = Mock()
        tool_response.choices[0].message.tool_calls = [tool_call]
        
        # Mock final response
        final_response = MockOpenAIResponse(content="Final answer after max rounds")
        
        # Configure mock to always want tools, then final response
        mock_client.chat.completions.create.side_effect = [
            tool_response,    # Round 1 tool call
            tool_response,    # Round 1 decision (wants more tools)
            tool_response,    # Round 2 tool call  
            final_response    # Final response (forced due to max rounds)
        ]
        mock_openai_class.return_value = mock_client
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "gpt-5", "http://localhost:4141/v1")
        
        tools = [{"name": "search_course_content", "description": "Search"}]
        
        # Test with max_rounds=2
        result = ai_gen.generate_response("Test query", tools=tools, tool_manager=mock_tool_manager, max_rounds=2)
        
        # Verify it terminated properly
        self.assertEqual(result, "Final answer after max rounds")
        
        # Verify tools were called exactly twice (once per round)
        self.assertEqual(mock_tool_manager.execute_tool.call_count, 2)

    @patch('ai_generator.OpenAI')
    def test_tool_execution_error_handling(self, mock_openai_class):
        """Test graceful handling of tool execution errors"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock tool manager that raises an error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # Create tool call
        tool_call = MockToolCall(
            id="call_123",
            function_name="search_course_content",
            arguments='{"query": "test"}'
        )
        
        # Mock response with tool calls
        initial_response = MockOpenAIResponse(
            content=None,
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )
        initial_response.choices[0].message = Mock()
        initial_response.choices[0].message.tool_calls = [tool_call]
        
        # Mock decision to stop after error (no more tool calls)
        decision_response = MockOpenAIResponse(content="Answer based on available information")
        
        # Mock final response (should not be reached in this case)
        final_response = MockOpenAIResponse(content="Final answer despite tool error")
        
        mock_client.chat.completions.create.side_effect = [
            initial_response,    # Tool call that will fail
            decision_response,   # Decision to stop (AI provides final answer)
            final_response       # Not reached
        ]
        mock_openai_class.return_value = mock_client
        
        # Create AI generator
        ai_gen = AIGenerator("test_key", "gpt-5", "http://localhost:4141/v1")
        
        tools = [{"name": "search_course_content", "description": "Search"}]
        
        # Should handle error gracefully and not raise exception
        result = ai_gen.generate_response("Test query", tools=tools, tool_manager=mock_tool_manager)
        
        # Verify it completed despite the error with the decision response
        self.assertEqual(result, "Answer based on available information")
        
        # Verify tool was attempted
        mock_tool_manager.execute_tool.assert_called_once()


if __name__ == "__main__":
    unittest.main()