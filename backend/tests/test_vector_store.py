import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import json

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults, OpenAIEmbeddingFunction
from models import Course, Lesson, CourseChunk


class TestSearchResults(unittest.TestCase):
    """Test SearchResults functionality"""

    def test_from_chroma_success(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'course': 'test1'}, {'course': 'test2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        self.assertEqual(results.documents, ['doc1', 'doc2'])
        self.assertEqual(results.metadata, [{'course': 'test1'}, {'course': 'test2'}])
        self.assertEqual(results.distances, [0.1, 0.2])
        self.assertIsNone(results.error)

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        self.assertEqual(results.documents, [])
        self.assertEqual(results.metadata, [])
        self.assertEqual(results.distances, [])
        self.assertTrue(results.is_empty())

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Test error message")
        
        self.assertEqual(results.documents, [])
        self.assertEqual(results.metadata, [])
        self.assertEqual(results.distances, [])
        self.assertEqual(results.error, "Test error message")
        self.assertTrue(results.is_empty())


class TestOpenAIEmbeddingFunction(unittest.TestCase):
    """Test OpenAI embedding function"""

    @patch('vector_store.OpenAI')
    def test_embedding_generation(self, mock_openai_class):
        """Test embedding generation"""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Create embedding function
        embedding_func = OpenAIEmbeddingFunction(
            model="text-embedding-3-small",
            api_key="test_key",
            base_url="http://localhost:4141/v1"
        )
        
        # Generate embeddings
        texts = ["text1", "text2"]
        embeddings = embedding_func(texts)
        
        # Verify results
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[1], [0.4, 0.5, 0.6])
        
        # Verify API call
        mock_client.embeddings.create.assert_called_once_with(
            input=texts,
            model="text-embedding-3-small"
        )


class TestVectorStore(unittest.TestCase):
    """Test VectorStore functionality"""

    def setUp(self):
        """Set up test fixtures"""
        with patch('vector_store.chromadb.PersistentClient'), \
             patch('vector_store.OpenAIEmbeddingFunction'):
            self.vector_store = VectorStore(
                chroma_path="./test_chroma",
                embedding_model="text-embedding-3-small",
                api_key="test_key",
                base_url="http://localhost:4141/v1",
                max_results=5
            )

    @patch('vector_store.chromadb.PersistentClient')
    def test_search_successful(self, mock_client_class):
        """Test successful search operation"""
        # Mock ChromaDB collections
        mock_collection = Mock()
        mock_collection.query.return_value = {
            'documents': [['Course content about Python']],
            'metadatas': [[{'course_title': 'Python Basics', 'lesson_number': 1}]],
            'distances': [[0.1]]
        }
        
        # Mock client
        mock_client = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        # Create vector store
        with patch('vector_store.OpenAIEmbeddingFunction'):
            vs = VectorStore("./test", "model", "key", "url")
            vs.course_content = mock_collection
        
        # Execute search
        results = vs.search("Python programming")
        
        # Verify results
        self.assertFalse(results.is_empty())
        self.assertEqual(len(results.documents), 1)
        self.assertIn("Python", results.documents[0])
        self.assertIsNone(results.error)

    @patch('vector_store.chromadb.PersistentClient')
    def test_search_with_course_filter(self, mock_client_class):
        """Test search with course name filtering"""
        # Mock course catalog for name resolution
        mock_catalog = Mock()
        mock_catalog.query.return_value = {
            'documents': [['Python Basics']],
            'metadatas': [[{'title': 'Python Basics Course'}]]
        }
        
        # Mock content collection
        mock_content = Mock()
        mock_content.query.return_value = {
            'documents': [['Filtered content']],
            'metadatas': [[{'course_title': 'Python Basics Course'}]],
            'distances': [[0.1]]
        }
        
        # Setup vector store
        with patch('vector_store.OpenAIEmbeddingFunction'):
            vs = VectorStore("./test", "model", "key", "url")
            vs.course_catalog = mock_catalog
            vs.course_content = mock_content
        
        # Execute search with course filter
        results = vs.search("Python", course_name="Python")
        
        # Verify course resolution was attempted
        mock_catalog.query.assert_called_once_with(
            query_texts=["Python"],
            n_results=1
        )
        
        # Verify content search was called with filter
        mock_content.query.assert_called_once()
        call_args = mock_content.query.call_args
        self.assertIn("where", call_args[1])

    @patch('vector_store.chromadb.PersistentClient')
    def test_search_course_not_found(self, mock_client_class):
        """Test search when course name cannot be resolved"""
        # Mock course catalog returning empty results
        mock_catalog = Mock()
        mock_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]]
        }
        
        with patch('vector_store.OpenAIEmbeddingFunction'):
            vs = VectorStore("./test", "model", "key", "url")
            vs.course_catalog = mock_catalog
        
        # Execute search with non-existent course
        results = vs.search("Python", course_name="NonexistentCourse")
        
        # Should return error
        self.assertTrue(results.is_empty())
        self.assertIn("No course found matching", results.error)

    @patch('vector_store.chromadb.PersistentClient')
    def test_search_exception_handling(self, mock_client_class):
        """Test search error handling"""
        # Mock collection that raises exception
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("ChromaDB connection failed")
        
        with patch('vector_store.OpenAIEmbeddingFunction'):
            vs = VectorStore("./test", "model", "key", "url")
            vs.course_content = mock_collection
        
        # Execute search
        results = vs.search("test query")
        
        # Should return error
        self.assertTrue(results.is_empty())
        self.assertIn("Search error", results.error)
        self.assertIn("ChromaDB connection failed", results.error)

    def test_build_filter_combinations(self):
        """Test filter building for different parameter combinations"""
        vs = self.vector_store
        
        # No filters
        filter_dict = vs._build_filter(None, None)
        self.assertIsNone(filter_dict)
        
        # Course only
        filter_dict = vs._build_filter("Python Course", None)
        self.assertEqual(filter_dict, {"course_title": "Python Course"})
        
        # Lesson only
        filter_dict = vs._build_filter(None, 1)
        self.assertEqual(filter_dict, {"lesson_number": 1})
        
        # Both course and lesson
        filter_dict = vs._build_filter("Python Course", 1)
        expected = {"$and": [
            {"course_title": "Python Course"},
            {"lesson_number": 1}
        ]}
        self.assertEqual(filter_dict, expected)

    @patch('vector_store.chromadb.PersistentClient')
    def test_add_course_metadata(self, mock_client_class):
        """Test adding course metadata"""
        # Mock collection
        mock_collection = Mock()
        
        with patch('vector_store.OpenAIEmbeddingFunction'):
            vs = VectorStore("./test", "model", "key", "url")
            vs.course_catalog = mock_collection
        
        # Create test course
        lessons = [
            Lesson(lesson_number=1, title="Intro", lesson_link="http://lesson1.com"),
            Lesson(lesson_number=2, title="Advanced", lesson_link="http://lesson2.com")
        ]
        course = Course(
            title="Test Course",
            instructor="Test Instructor",
            course_link="http://course.com",
            lessons=lessons
        )
        
        # Add course metadata
        vs.add_course_metadata(course)
        
        # Verify collection.add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        # Verify document content
        self.assertEqual(call_args[1]["documents"], ["Test Course"])
        self.assertEqual(call_args[1]["ids"], ["Test Course"])
        
        # Verify metadata structure
        metadata = call_args[1]["metadatas"][0]
        self.assertEqual(metadata["title"], "Test Course")
        self.assertEqual(metadata["instructor"], "Test Instructor")
        self.assertEqual(metadata["lesson_count"], 2)
        
        # Verify lessons JSON
        lessons_json = json.loads(metadata["lessons_json"])
        self.assertEqual(len(lessons_json), 2)
        self.assertEqual(lessons_json[0]["lesson_number"], 1)

    @patch('vector_store.chromadb.PersistentClient')
    def test_add_course_content(self, mock_client_class):
        """Test adding course content chunks"""
        # Mock collection
        mock_collection = Mock()
        
        with patch('vector_store.OpenAIEmbeddingFunction'):
            vs = VectorStore("./test", "model", "key", "url")
            vs.course_content = mock_collection
        
        # Create test chunks
        chunks = [
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=0,
                content="First chunk content"
            ),
            CourseChunk(
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1,
                content="Second chunk content"
            )
        ]
        
        # Add content
        vs.add_course_content(chunks)
        
        # Verify collection.add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        
        # Verify documents
        expected_docs = ["First chunk content", "Second chunk content"]
        self.assertEqual(call_args[1]["documents"], expected_docs)
        
        # Verify IDs
        expected_ids = ["Test_Course_0", "Test_Course_1"]
        self.assertEqual(call_args[1]["ids"], expected_ids)
        
        # Verify metadata
        metadata = call_args[1]["metadatas"]
        self.assertEqual(len(metadata), 2)
        self.assertEqual(metadata[0]["course_title"], "Test Course")
        self.assertEqual(metadata[0]["lesson_number"], 1)
        self.assertEqual(metadata[0]["chunk_index"], 0)

    @patch('vector_store.chromadb.PersistentClient')
    def test_get_existing_course_titles(self, mock_client_class):
        """Test getting existing course titles"""
        # Mock collection
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        
        with patch('vector_store.OpenAIEmbeddingFunction'):
            vs = VectorStore("./test", "model", "key", "url")
            vs.course_catalog = mock_collection
        
        # Get titles
        titles = vs.get_existing_course_titles()
        
        # Verify results
        self.assertEqual(titles, ['Course 1', 'Course 2', 'Course 3'])

    @patch('vector_store.chromadb.PersistentClient')
    def test_get_course_count(self, mock_client_class):
        """Test getting course count"""
        # Mock collection
        mock_collection = Mock()
        mock_collection.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }
        
        with patch('vector_store.OpenAIEmbeddingFunction'):
            vs = VectorStore("./test", "model", "key", "url")
            vs.course_catalog = mock_collection
        
        # Get count
        count = vs.get_course_count()
        
        # Verify result
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()