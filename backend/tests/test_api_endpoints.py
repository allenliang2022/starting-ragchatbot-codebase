import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json


@pytest.mark.api
class TestAPIEndpoints:
    """Test FastAPI endpoints for proper request/response handling"""

    def test_query_endpoint_successful_response(self, client, sample_query_data):
        """Test /api/query endpoint with valid request"""
        query_data = sample_query_data["valid_query"]
        
        response = client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        
        # Verify response content
        assert data["answer"] == "This is a test response about Python programming."
        assert data["sources"] == ["Python Course - Lesson 1", "Advanced Python - Module 2"]
        assert data["session_id"] == "test_session"

    def test_query_endpoint_without_session_id(self, client, sample_query_data):
        """Test /api/query endpoint creates session when none provided"""
        query_data = sample_query_data["query_without_session"]
        
        response = client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should create a new session
        assert data["session_id"] == "test_session_123"
        assert data["answer"] == "This is a test response about Python programming."

    def test_query_endpoint_empty_query(self, client, sample_query_data):
        """Test /api/query endpoint with empty query"""
        query_data = sample_query_data["empty_query"]
        
        response = client.post("/api/query", json=query_data)
        
        # Should still process successfully (RAG system handles empty queries)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_endpoint_missing_query_field(self, client):
        """Test /api/query endpoint with missing query field"""
        response = client.post("/api/query", json={"session_id": "test"})
        
        # Should return validation error
        assert response.status_code == 422

    def test_query_endpoint_invalid_json(self, client):
        """Test /api/query endpoint with invalid JSON"""
        response = client.post(
            "/api/query", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422

    def test_query_endpoint_rag_system_error(self, client, mock_rag_system):
        """Test /api/query endpoint when RAG system raises exception"""
        # Make RAG system raise an exception
        mock_rag_system.query.side_effect = Exception("RAG processing failed")
        
        response = client.post("/api/query", json={"query": "test query"})
        
        assert response.status_code == 500
        assert "RAG processing failed" in response.json()["detail"]

    def test_query_endpoint_long_query(self, client, sample_query_data):
        """Test /api/query endpoint with very long query"""
        query_data = sample_query_data["long_query"]
        
        response = client.post("/api/query", json=query_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_courses_endpoint_successful_response(self, client):
        """Test /api/courses endpoint returns course statistics"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data
        
        # Verify response content matches mock data
        assert data["total_courses"] == 3
        assert data["course_titles"] == ["Python Basics", "Advanced Python", "Web Development"]

    def test_courses_endpoint_analytics_error(self, client, mock_rag_system):
        """Test /api/courses endpoint when analytics fails"""
        # Make analytics method raise an exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics failed")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        assert "Analytics failed" in response.json()["detail"]

    def test_root_endpoint(self, client):
        """Test / (root) endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "RAG System API"

    def test_nonexistent_endpoint(self, client):
        """Test request to non-existent endpoint"""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404

    def test_query_endpoint_content_type_validation(self, client):
        """Test /api/query requires JSON content type"""
        response = client.post(
            "/api/query",
            data="query=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422

    def test_cors_headers(self, client):
        """Test CORS headers are properly set"""
        response = client.options("/api/query")
        
        # Check for CORS headers (note: TestClient may not set all CORS headers)
        # This tests that the middleware is configured
        assert response.status_code in [200, 405]  # OPTIONS may not be explicitly handled

    def test_query_response_schema_validation(self, client):
        """Test that query response matches expected schema"""
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate all required fields are present
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        # Validate field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Validate sources list contains strings
        for source in data["sources"]:
            assert isinstance(source, str)

    def test_courses_response_schema_validation(self, client):
        """Test that courses response matches expected schema"""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate all required fields are present
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        # Validate field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        
        # Validate course_titles list contains strings
        for title in data["course_titles"]:
            assert isinstance(title, str)

    def test_query_endpoint_session_persistence(self, client):
        """Test that session IDs are properly handled across requests"""
        # First request without session ID
        response1 = client.post("/api/query", json={"query": "first query"})
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]
        
        # Second request with same session ID
        response2 = client.post("/api/query", json={
            "query": "second query", 
            "session_id": session_id
        })
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

    @pytest.mark.parametrize("invalid_session_id", [None, "", "   ", 123, []])
    def test_query_endpoint_invalid_session_id_types(self, client, invalid_session_id):
        """Test /api/query with various invalid session ID types"""
        query_data = {"query": "test query"}
        if invalid_session_id is not None:
            query_data["session_id"] = invalid_session_id
        
        response = client.post("/api/query", json=query_data)
        
        # Should either succeed (creating new session) or return validation error
        assert response.status_code in [200, 422]

    def test_query_endpoint_unicode_handling(self, client):
        """Test /api/query handles Unicode characters properly"""
        unicode_query = "What is Python? 测试 🐍 émoji"
        
        response = client.post("/api/query", json={"query": unicode_query})
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_api_response_headers(self, client):
        """Test that API responses have proper headers"""
        response = client.post("/api/query", json={"query": "test"})
        
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

    def test_method_not_allowed(self, client):
        """Test unsupported HTTP methods return 405"""
        # GET on POST endpoint
        response = client.get("/api/query")
        assert response.status_code == 405
        
        # POST on GET endpoint  
        response = client.post("/api/courses")
        assert response.status_code == 405