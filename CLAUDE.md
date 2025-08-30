# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for querying course materials. The application serves a FastAPI backend with a simple HTML/JS frontend, using ChromaDB for vector storage and DeepSeek for AI generation.

## Development Commands

**IMPORTANT: Always use `uv` for all Python operations - never use pip directly.**

### Start the application

#### Windows (Command Prompt/PowerShell)
```cmd
run.bat
```


### Manual backend start
```cmd
cd backend
uv run uvicorn app:app --reload --port 8000
```

### Install dependencies
```cmd
uv sync
```

### Add new dependencies
```cmd
uv add package-name
```

### Run Python files
```cmd
uv run python filename.py
```

## Architecture

### Core Components
- **FastAPI Backend** (`backend/app.py`): Main application server with CORS enabled, serves both API and static files
- **RAG System** (`backend/rag_system.py`): Central orchestrator coordinating all components
- **Vector Store** (`backend/vector_store.py`): ChromaDB interface for semantic search
- **Document Processor** (`backend/document_processor.py`): Handles text chunking and course parsing
- **AI Generator** (`backend/ai_generator.py`): DeepSeek API integration
- **Session Manager** (`backend/session_manager.py`): Manages conversation history
- **Search Tools** (`backend/search_tools.py`): Tool-based search functionality

### Data Models
All data models are defined in `backend/models.py` including Course, Lesson, and CourseChunk.

### Configuration
System configuration is centralized in `backend/config.py` using dataclasses and environment variables.

### Frontend
Simple static files in `frontend/` directory served directly by FastAPI.

## Environment Setup

Create a `.env` file based on `.env.example`:
```
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

## Key Configuration

- **Chunk Size**: 800 characters with 100 character overlap
- **Embedding Model**: text-embedding-3-small
- **Max Search Results**: 5
- **Conversation History**: 2 messages
- **DeepSeek Model**: deepseek-chat
- **ChromaDB Path**: ./chroma_db

## API Endpoints

- `POST /api/query`: Process user queries with RAG
- `GET /api/courses`: Get course statistics
- `GET /`: Serves frontend application

## Document Processing

Course documents are loaded from `docs/` directory on startup. The system automatically processes `.txt` files as course transcripts.
- always start the server using run.bat in a windows shell, remember you are on windows