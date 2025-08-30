# Copilot Instructions for Course Materials RAG System

## Architecture Overview
Full-stack RAG chatbot: FastAPI backend + vanilla JS frontend. Uses ChromaDB for semantic search with DeepSeek AI tool calling.

**Data Flow**: Course docs → `DocumentProcessor` → chunks → `VectorStore` (ChromaDB) → semantic search → DeepSeek with tools → response

## Critical Components (`backend/`)
- **`rag_system.py`**: Central orchestrator coordinating all components  
- **`vector_store.py`**: Dual ChromaDB collections - `course_catalog` + `course_content`
- **`search_tools.py`**: Tool interface for DeepSeek to search semantically
- **`ai_generator.py`**: DeepSeek API with tool calling (`gpt-5` model)
- **`document_processor.py`**: Sentence-aware chunking (800 chars, 100 overlap)

## Development Workflows

**CRITICAL: Always use `uv` - never pip directly**
```bash
# Start application (Windows)
run.bat  # or: cd backend && uv run uvicorn app:app --reload --port 8000

# Start application (Unix)
./run.sh

# Dependencies  
uv sync              # Install
uv add package-name  # Add new
uv run python file.py # Execute
```

**Environment**: Requires `DEEPSEEK_API_KEY` in `.env` (NOT Anthropic)

**Documents**: Place `.txt` files in `docs/` → auto-loaded on startup → check `/api/courses`

## Key Patterns

**Tool-Based AI**: DeepSeek calls `search_course_content(query, course_name, lesson_number)` via `search_tools.py`

**Two-Stage Search**: 
1. Course resolution (fuzzy name matching)  
2. Semantic content search with lesson filtering

**Session Management**: Auto-created sessions with 2-message history via `session_manager.py`

**Data Models** (`models.py`): `Course` (title=ID), `CourseChunk`, `Lesson`

## API Endpoints
- `POST /api/query`: Chat with RAG (returns answer + sources + session_id)
- `GET /api/courses`: Course analytics

## Configuration (`config.py`)
- Embedding: `text-embedding-3-small` (OpenAI compatible via DeepSeek)
- AI Model: `gpt-5` via DeepSeek API (`localhost:4141`)  
- ChromaDB: `./chroma_db`
- Chunking: 800 chars, 100 overlap

## Common Tasks
**Add content**: Drop `.txt` in `docs/` → restart → verify at `/api/courses`
**Tune search**: Modify `CHUNK_SIZE`/`MAX_RESULTS` in `config.py`
**Debug**: Check `vector_store.py:search()` and `ai_generator.py:SYSTEM_PROMPT`
**Frontend**: Vanilla JS in `frontend/` - no build step required
