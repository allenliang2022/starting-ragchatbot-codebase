import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool(unittest.TestCase):
    """Test CourseSearchTool execute method and error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_vector_store = Mock()
        self.search_tool = CourseSearchTool(self.mock_vector_store)

    def test_execute_successful_search(self):
        """Test successful search with results"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["Course content about Python", "More Python content"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search
        result = self.search_tool.execute(
            query="Python basics", course_name="Python", lesson_number=1
        )

        # Verify search was called with correct parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="Python basics", course_name="Python", lesson_number=1
        )

        # Verify formatted results
        self.assertIn("[Python Basics - Lesson 1]", result)
        self.assertIn("Course content about Python", result)
        self.assertEqual(
            self.search_tool.last_sources,
            ["Python Basics - Lesson 1", "Python Basics - Lesson 2"],
        )

    def test_execute_with_error(self):
        """Test search tool when vector store returns error"""
        # Mock error result
        mock_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Vector store connection failed",
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search
        result = self.search_tool.execute(query="test query")

        # Should return the error message
        self.assertEqual(result, "Vector store connection failed")

    def test_execute_empty_results(self):
        """Test search tool when no results found"""
        # Mock empty results
        mock_results = SearchResults(
            documents=[], metadata=[], distances=[], error=None
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search with course filter
        result = self.search_tool.execute(
            query="nonexistent topic", course_name="Python Course"
        )

        # Should return no results message with filter info
        self.assertIn("No relevant content found in course 'Python Course'", result)

    def test_execute_query_only(self):
        """Test search with only query parameter"""
        # Mock successful search results
        mock_results = SearchResults(
            documents=["General content"],
            metadata=[{"course_title": "General Course", "lesson_number": None}],
            distances=[0.1],
            error=None,
        )
        self.mock_vector_store.search.return_value = mock_results

        # Execute search with only query
        result = self.search_tool.execute(query="general topic")

        # Verify search was called with correct parameters
        self.mock_vector_store.search.assert_called_once_with(
            query="general topic", course_name=None, lesson_number=None
        )

        # Verify formatted results
        self.assertIn("[General Course]", result)
        self.assertIn("General content", result)

    def test_format_results_unique_sources(self):
        """Test that duplicate sources are removed while preserving order"""
        # Mock results with duplicate sources
        mock_results = SearchResults(
            documents=["Content 1", "Content 2", "Content 3"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course A", "lesson_number": 1},  # Duplicate
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.15, 0.2],
            error=None,
        )

        # Call the private method directly
        result = self.search_tool._format_results(mock_results)

        # Verify unique sources
        expected_sources = ["Course A - Lesson 1", "Course B - Lesson 2"]
        self.assertEqual(self.search_tool.last_sources, expected_sources)

    def test_get_tool_definition(self):
        """Test tool definition format"""
        definition = self.search_tool.get_tool_definition()

        # Verify structure
        self.assertEqual(definition["name"], "search_course_content")
        self.assertIn("description", definition)
        self.assertIn("input_schema", definition)

        # Verify required parameters
        required = definition["input_schema"]["required"]
        self.assertEqual(required, ["query"])


class TestToolManager(unittest.TestCase):
    """Test ToolManager functionality"""

    def setUp(self):
        self.tool_manager = ToolManager()
        self.mock_tool = Mock()
        self.mock_tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool",
        }

    def test_register_tool(self):
        """Test tool registration"""
        self.tool_manager.register_tool(self.mock_tool)

        # Verify tool is registered
        self.assertIn("test_tool", self.tool_manager.tools)
        self.assertEqual(self.tool_manager.tools["test_tool"], self.mock_tool)

    def test_execute_tool(self):
        """Test tool execution"""
        self.mock_tool.execute.return_value = "Test result"
        self.tool_manager.register_tool(self.mock_tool)

        # Execute tool
        result = self.tool_manager.execute_tool("test_tool", param1="value1")

        # Verify execution
        self.mock_tool.execute.assert_called_once_with(param1="value1")
        self.assertEqual(result, "Test result")

    def test_execute_nonexistent_tool(self):
        """Test executing tool that doesn't exist"""
        result = self.tool_manager.execute_tool("nonexistent_tool")

        self.assertEqual(result, "Tool 'nonexistent_tool' not found")

    def test_get_last_sources(self):
        """Test getting sources from tools"""
        # Mock tool with sources
        mock_tool_with_sources = Mock()
        mock_tool_with_sources.last_sources = ["Source 1", "Source 2"]
        mock_tool_with_sources.get_tool_definition.return_value = {
            "name": "tool_with_sources"
        }

        self.tool_manager.register_tool(mock_tool_with_sources)

        # Get sources
        sources = self.tool_manager.get_last_sources()
        self.assertEqual(sources, ["Source 1", "Source 2"])


if __name__ == "__main__":
    unittest.main()
