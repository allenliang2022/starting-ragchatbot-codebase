import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be.*")

import logging
import os
import traceback
from typing import List, Optional

from config import config
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_system import RAGSystem

# Initialize FastAPI app with debug mode
app = FastAPI(title="Course Materials RAG System", root_path="", debug=True)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add trusted host middleware for proxy
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Enable CORS with proper settings for proxy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Initialize RAG system
rag_system = RAGSystem(config)


# Global exception handler for detailed error logging
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions and log detailed information"""
    error_traceback = traceback.format_exc()
    logger.error(f"Unhandled exception at {request.url}:\n{error_traceback}")
    print(f"ERROR: Unhandled exception at {request.url}:\n{error_traceback}")
    return HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Request model for course queries"""

    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for course queries"""

    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    """Response model for course statistics"""

    total_courses: int
    course_titles: List[str]


# API Endpoints


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Process a query and return response with sources"""
    try:
        logger.info(f"Processing query: {request.query}")
        print(f"DEBUG: Processing query: {request.query}")

        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            logger.debug("Creating new session")
            session_id = rag_system.session_manager.create_session()
            print(f"DEBUG: Created session: {session_id}")

        # Process query using RAG system
        logger.debug(
            f"Calling RAG system with query: {request.query}, session: {session_id}"
        )
        print(f"DEBUG: About to call rag_system.query()")
        answer, sources = rag_system.query(request.query, session_id)
        print(
            f"DEBUG: RAG system returned answer: {answer[:100]}..., sources: {sources}"
        )

        return QueryResponse(answer=answer, sources=sources, session_id=session_id)
    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"Error in query_documents: {error_traceback}")
        print(f"ERROR in query_documents: {error_traceback}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")


@app.get("/api/courses", response_model=CourseStats)
async def get_course_stats():
    """Get course analytics and statistics"""
    try:
        analytics = rag_system.get_course_analytics()
        return CourseStats(
            total_courses=analytics["total_courses"],
            course_titles=analytics["course_titles"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Load initial documents on startup"""
    docs_path = "../docs"
    if os.path.exists(docs_path):
        print("Loading initial documents...")
        try:
            courses, chunks = rag_system.add_course_folder(
                docs_path, clear_existing=False
            )
            print(f"Loaded {courses} courses with {chunks} chunks")
        except Exception as e:
            print(f"Error loading documents: {e}")


import os
from pathlib import Path

from fastapi.responses import FileResponse

# Custom static file handler with no-cache headers for development
from fastapi.staticfiles import StaticFiles


class DevStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            # Add no-cache headers for development
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


# Serve static files for the frontend
app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
