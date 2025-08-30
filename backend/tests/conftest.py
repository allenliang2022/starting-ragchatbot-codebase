import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from fastapi import FastAPI
from config import Config
from rag_system import RAGSystem


@pytest.fixture
def test_config():
    """Create a test configuration"""
    config = Config()
    config.DEEPSEEK_API_KEY = "test_key"
    config.CHROMA_PATH = "./test_chroma"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_SEARCH_RESULTS = 5
    config.MAX_CONVERSATION_HISTORY = 2
    return config


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_rag_system():
    """Create a mocked RAG system for API testing"""
    mock_rag = Mock(spec=RAGSystem)
    
    # Mock session manager
    mock_session_manager = Mock()
    mock_session_manager.create_session.return_value = "test_session_123"
    mock_rag.session_manager = mock_session_manager
    
    # Mock query method
    mock_rag.query.return_value = (
        "This is a test response about Python programming.",
        ["Python Course - Lesson 1", "Advanced Python - Module 2"]
    )
    
    # Mock analytics method
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Python Basics", "Advanced Python", "Web Development"]
    }
    
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional
    
    # Create test app
    app = FastAPI(title="Test RAG System")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None
    
    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str
    
    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # API endpoints (inline to avoid import issues)
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def root():
        return {"message": "RAG System API"}
    
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def sample_query_data():
    """Sample data for query testing"""
    return {
        "valid_query": {
            "query": "What is Python?",
            "session_id": "test_session"
        },
        "query_without_session": {
            "query": "Explain variables in Python"
        },
        "empty_query": {
            "query": ""
        },
        "long_query": {
            "query": "A" * 1000
        }
    }


@pytest.fixture
def sample_course_data():
    """Sample course data for testing"""
    return {
        "course_titles": ["Python Basics", "Advanced Python", "Web Development"],
        "total_courses": 3,
        "chunks": [
            {
                "content": "Python is a programming language",
                "source": "Python Course - Lesson 1",
                "course_title": "Python Basics"
            },
            {
                "content": "Variables store data in Python",
                "source": "Python Course - Lesson 2", 
                "course_title": "Python Basics"
            }
        ]
    }


@pytest.fixture
def mock_ai_generator():
    """Create a mocked AI generator"""
    mock_ai = Mock()
    mock_ai.generate_response.return_value = "Generated AI response"
    return mock_ai


@pytest.fixture
def mock_vector_store():
    """Create a mocked vector store"""
    mock_store = Mock()
    mock_store.get_course_count.return_value = 3
    mock_store.get_existing_course_titles.return_value = [
        "Python Basics", "Advanced Python", "Web Development"
    ]
    mock_store.search.return_value = [
        {"content": "Test content", "metadata": {"source": "Test source"}}
    ]
    return mock_store


@pytest.fixture
def mock_session_manager():
    """Create a mocked session manager"""
    mock_session = Mock()
    mock_session.create_session.return_value = "test_session_123"
    mock_session.get_conversation_history.return_value = "Previous conversation"
    return mock_session


@pytest.fixture
def mock_tool_manager():
    """Create a mocked tool manager"""
    mock_tools = Mock()
    mock_tools.get_tool_definitions.return_value = [
        {"name": "search_course_content", "description": "Search course content"}
    ]
    mock_tools.get_last_sources.return_value = ["Test Source 1", "Test Source 2"]
    return mock_tools


@pytest.fixture(autouse=True)
def patch_external_dependencies():
    """Auto-patch external dependencies that might cause issues in tests"""
    with patch('chromadb.PersistentClient'), \
         patch('openai.OpenAI'), \
         patch('sentence_transformers.SentenceTransformer'):
        yield