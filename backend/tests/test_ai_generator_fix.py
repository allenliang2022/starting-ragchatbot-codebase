import json
import os
import sys
import unittest
from unittest.mock import Mock, patch

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


class TestAIGeneratorFix(unittest.TestCase):
    """Test that the AI generator fix works correctly"""

    @patch("ai_generator.OpenAI")
    def test_fixed_json_parsing_with_booleans(self, mock_openai_class):
        """Test that the json.loads fix handles JSON with booleans correctly"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool executed successfully"

        # Create tool call with JSON containing booleans (this used to break eval())
        tool_call = MockToolCall(
            id="call_123",
            function_name="search_course_content",
            arguments='{"query": "Python basics", "enabled": true, "data": null}',
        )

        # Mock initial response with tool calls
        initial_response = MockOpenAIResponse(
            content=None, tool_calls=[tool_call], finish_reason="tool_calls"
        )
        initial_response.choices[0].message = Mock()
        initial_response.choices[0].message.tool_calls = [tool_call]

        # Mock final response
        final_response = MockOpenAIResponse(content="Final response with tool results")

        # Configure mock to return different responses
        mock_client.chat.completions.create.side_effect = [
            initial_response,
            final_response,
        ]

        # Create AI generator
        ai_gen = AIGenerator("test_key", "gpt-5", "http://localhost:4141/v1")

        # Mock tools
        tools = [
            {
                "name": "search_course_content",
                "description": "Search for course content",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]

        # This should now work with the json.loads fix!
        result = ai_gen.generate_response(
            "Test query", tools=tools, tool_manager=mock_tool_manager
        )

        # Verify it completed successfully
        self.assertEqual(result, "Final response with tool results")

        # Verify tool was called with correctly parsed arguments
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics",
            enabled=True,  # Note: JSON true became Python True
            data=None,  # Note: JSON null became Python None
        )

    @patch("ai_generator.OpenAI")
    def test_fix_handles_malformed_json_gracefully(self, mock_openai_class):
        """Test that malformed JSON is handled gracefully"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()

        # Create tool call with malformed JSON
        tool_call = MockToolCall(
            id="call_123",
            function_name="search_course_content",
            arguments='{"query": "incomplete',  # Malformed JSON
        )

        # Mock response with tool calls
        initial_response = MockOpenAIResponse(
            content=None, tool_calls=[tool_call], finish_reason="tool_calls"
        )
        initial_response.choices[0].message = Mock()
        initial_response.choices[0].message.tool_calls = [tool_call]

        # Mock decision response after error
        decision_response = MockOpenAIResponse(content="Error handling response")

        mock_client.chat.completions.create.side_effect = [
            initial_response,
            decision_response,
        ]

        # Create AI generator
        ai_gen = AIGenerator("test_key", "gpt-5", "http://localhost:4141/v1")

        tools = [{"name": "search_course_content", "description": "Search"}]

        # Should handle malformed JSON gracefully and not raise exception
        # The new implementation handles JSON errors gracefully by adding error message to conversation
        try:
            result = ai_gen.generate_response(
                "Test query", tools=tools, tool_manager=mock_tool_manager
            )
            # Should get a response despite the JSON error
            self.assertEqual(result, "Error handling response")
        except json.JSONDecodeError:
            self.fail("JSONDecodeError should be handled gracefully, not raised")


if __name__ == "__main__":
    unittest.main()
