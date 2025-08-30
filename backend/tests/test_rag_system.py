import os
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from rag_system import RAGSystem
from search_tools import CourseSearchTool, ToolManager


class TestRAGSystem(unittest.TestCase):
    """Test RAG system end-to-end query processing"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a test config
        self.test_config = Config()
        self.test_config.DEEPSEEK_API_KEY = "test_key"
        self.test_config.CHROMA_PATH = "./test_chroma"

        # Mock all external dependencies
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.VectorStore"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):

            self.rag_system = RAGSystem(self.test_config)

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_query_successful_response(
        self, mock_ai_generator_class, mock_vector_store_class
    ):
        """Test successful query processing"""
        # Mock AI generator
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = (
            "This is a test response about Python."
        )
        mock_ai_generator_class.return_value = mock_ai_gen

        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search content"}
        ]
        mock_tool_manager.get_last_sources.return_value = ["Python Course - Lesson 1"]

        # Create RAG system with mocked components
        with patch("rag_system.DocumentProcessor"), patch("rag_system.SessionManager"):
            rag_sys = RAGSystem(self.test_config)
            rag_sys.ai_generator = mock_ai_gen
            rag_sys.tool_manager = mock_tool_manager

        # Execute query
        response, sources = rag_sys.query("What is Python?")

        # Verify response
        self.assertEqual(response, "This is a test response about Python.")
        self.assertEqual(sources, ["Python Course - Lesson 1"])

        # Verify AI generator was called with correct parameters
        mock_ai_gen.generate_response.assert_called_once()
        call_args = mock_ai_gen.generate_response.call_args
        self.assertIn("What is Python?", call_args.kwargs["query"])
        self.assertEqual(
            call_args.kwargs["tools"],
            [{"name": "search_course_content", "description": "Search content"}],
        )
        self.assertEqual(call_args.kwargs["tool_manager"], mock_tool_manager)

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_query_with_session_management(
        self, mock_ai_generator_class, mock_vector_store_class
    ):
        """Test query processing with session management"""
        # Mock AI generator
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Response with context"
        mock_ai_generator_class.return_value = mock_ai_gen

        # Mock session manager
        mock_session_manager = Mock()
        mock_session_manager.get_conversation_history.return_value = (
            "Previous: Hello\nAssistant: Hi there!"
        )

        # Mock other components
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = []
        mock_tool_manager.get_last_sources.return_value = []

        # Create RAG system
        with patch("rag_system.DocumentProcessor"):
            rag_sys = RAGSystem(self.test_config)
            rag_sys.ai_generator = mock_ai_gen
            rag_sys.session_manager = mock_session_manager
            rag_sys.tool_manager = mock_tool_manager

        # Execute query with session
        response, sources = rag_sys.query(
            "Follow up question", session_id="test_session"
        )

        # Verify session management
        mock_session_manager.get_conversation_history.assert_called_once_with(
            "test_session"
        )

        # Verify AI generator received conversation history
        call_args = mock_ai_gen.generate_response.call_args
        self.assertEqual(
            call_args.kwargs["conversation_history"],
            "Previous: Hello\nAssistant: Hi there!",
        )

        # Verify session was updated
        mock_session_manager.add_exchange.assert_called_once_with(
            "test_session", "Follow up question", "Response with context"
        )

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_query_ai_generator_failure(
        self, mock_ai_generator_class, mock_vector_store_class
    ):
        """Test query when AI generator fails"""
        # Mock AI generator to raise exception
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.side_effect = Exception("AI generation failed")
        mock_ai_generator_class.return_value = mock_ai_gen

        # Mock other components
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = []

        # Create RAG system
        with patch("rag_system.DocumentProcessor"), patch("rag_system.SessionManager"):
            rag_sys = RAGSystem(self.test_config)
            rag_sys.ai_generator = mock_ai_gen
            rag_sys.tool_manager = mock_tool_manager

        # Execute query - should propagate exception
        with self.assertRaises(Exception) as context:
            rag_sys.query("Test query")

        self.assertIn("AI generation failed", str(context.exception))

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_query_tool_execution_failure(
        self, mock_ai_generator_class, mock_vector_store_class
    ):
        """Test query when tool execution fails within AI generator"""
        # Mock AI generator that simulates the eval() bug
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.side_effect = SyntaxError(
            "invalid syntax (eval of JSON)"
        )
        mock_ai_generator_class.return_value = mock_ai_gen

        # Mock other components
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [
            {"name": "search_course_content", "description": "Search"}
        ]

        # Create RAG system
        with patch("rag_system.DocumentProcessor"), patch("rag_system.SessionManager"):
            rag_sys = RAGSystem(self.test_config)
            rag_sys.ai_generator = mock_ai_gen
            rag_sys.tool_manager = mock_tool_manager

        # Execute query - should propagate the eval syntax error
        with self.assertRaises(SyntaxError) as context:
            rag_sys.query("What is Python?")

        self.assertIn("invalid syntax", str(context.exception))

    def test_initialization_components(self):
        """Test that RAG system initializes all required components"""
        # Verify components are initialized
        self.assertIsNotNone(self.rag_system.document_processor)
        self.assertIsNotNone(self.rag_system.vector_store)
        self.assertIsNotNone(self.rag_system.ai_generator)
        self.assertIsNotNone(self.rag_system.session_manager)
        self.assertIsNotNone(self.rag_system.tool_manager)
        self.assertIsNotNone(self.rag_system.search_tool)
        self.assertIsNotNone(self.rag_system.outline_tool)

    @patch("rag_system.VectorStore")
    def test_get_course_analytics(self, mock_vector_store_class):
        """Test course analytics functionality"""
        # Mock vector store
        mock_vector_store = Mock()
        mock_vector_store.get_course_count.return_value = 3
        mock_vector_store.get_existing_course_titles.return_value = [
            "Python Basics",
            "Advanced Python",
            "Web Development",
        ]
        mock_vector_store_class.return_value = mock_vector_store

        # Create RAG system
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.AIGenerator"),
            patch("rag_system.SessionManager"),
        ):
            rag_sys = RAGSystem(self.test_config)
            rag_sys.vector_store = mock_vector_store

        # Get analytics
        analytics = rag_sys.get_course_analytics()

        # Verify analytics
        self.assertEqual(analytics["total_courses"], 3)
        self.assertEqual(len(analytics["course_titles"]), 3)
        self.assertIn("Python Basics", analytics["course_titles"])

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    def test_source_tracking_and_reset(
        self, mock_ai_generator_class, mock_vector_store_class
    ):
        """Test that sources are properly tracked and reset"""
        # Mock AI generator
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Test response"
        mock_ai_generator_class.return_value = mock_ai_gen

        # Mock tool manager with sources
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = []
        mock_tool_manager.get_last_sources.return_value = ["Source 1", "Source 2"]

        # Create RAG system
        with (
            patch("rag_system.DocumentProcessor"),
            patch("rag_system.SessionManager"),
            patch("rag_system.VectorStore"),
        ):
            rag_sys = RAGSystem(self.test_config)
            rag_sys.ai_generator = mock_ai_gen
            rag_sys.tool_manager = mock_tool_manager

        # Execute query
        response, sources = rag_sys.query("Test query")

        # Verify sources were retrieved and reset was called
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.reset_sources.assert_called_once()
        self.assertEqual(sources, ["Source 1", "Source 2"])


if __name__ == "__main__":
    unittest.main()
