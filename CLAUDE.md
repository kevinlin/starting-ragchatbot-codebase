# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start using the provided script
./run.sh

# Manual start (from root directory)
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Setup
```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Package Management
This project uses **uv** as the Python package manager. All dependencies are defined in `pyproject.toml`.

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with a layered architecture:

### Core RAG Pipeline (backend/rag_system.py)
The `RAGSystem` class orchestrates the entire pipeline:
1. **Document Processing** → **Vector Storage** → **AI Generation**
2. Uses **tool-based search** where Claude dynamically calls search tools rather than pre-retrieving context
3. Maintains conversation sessions for multi-turn interactions

### Vector Storage Strategy (backend/vector_store.py)
Two-collection approach in ChromaDB:
- **course_catalog**: Course metadata (titles, instructors) for semantic course resolution
- **course_content**: Actual course content chunks for content search

The search flow: Query → Resolve course name via semantic search → Filter and search content → Return results

### AI Integration (backend/ai_generator.py)
- Uses Anthropic Claude with **function calling** for tool-based retrieval
- Tools are dynamically invoked during conversation rather than pre-retrieval
- Maintains conversation history via `SessionManager`

### Document Processing (backend/document_processor.py)
- Parses course documents (PDF, DOCX, TXT) from the `docs/` folder
- Extracts structured metadata (course title, instructor, lessons)
- Chunks content with configurable overlap for vector storage

### API Layer (backend/app.py)
FastAPI server with two main endpoints:
- `POST /api/query`: Main chat interface with session management
- `GET /api/courses`: Course analytics and metadata

### Frontend Architecture
Vanilla JavaScript SPA (`frontend/script.js`) that:
- Manages chat sessions and conversation state
- Handles real-time loading states and markdown rendering
- Displays course statistics and suggested questions

## Key Configuration (backend/config.py)

Critical settings that affect system behavior:
- `CHUNK_SIZE: 800` / `CHUNK_OVERLAP: 100`: Controls document chunking granularity
- `MAX_RESULTS: 5`: Limits search results per query
- `MAX_HISTORY: 2`: Conversation context window size
- `ANTHROPIC_MODEL`: Currently set to "claude-sonnet-4-20250514"

## Data Flow and Storage

### Document Ingestion
1. Documents in `docs/` are processed on startup via `app.py:startup_event()`
2. Course metadata extracted and stored in `course_catalog` collection
3. Content chunked and stored in `course_content` collection
4. Existing courses are skipped to avoid duplication

### Query Processing
1. User query sent to `/api/query` endpoint
2. `RAGSystem.query()` creates prompt for Claude with tool definitions
3. Claude dynamically calls `CourseSearchTool` as needed during response generation
4. Search tool uses semantic similarity to find relevant content
5. Response generated with retrieved context and conversation history

### Session Management
- Sessions created automatically for new conversations
- Conversation history maintained in memory (configurable length)
- Session IDs passed between frontend and backend

## Search Tools Architecture (backend/search_tools.py)

The system uses a **tool-based approach** rather than traditional RAG pre-retrieval:
- `ToolManager` registers available search tools
- `CourseSearchTool` provides semantic search capabilities to Claude
- Tools are called dynamically during AI response generation
- This allows Claude to make multiple searches or refine queries as needed

## Frontend-Backend Integration

The frontend uses relative API paths (`/api`) and expects:
- JSON responses with `answer`, `sources`, and `session_id` fields
- Session persistence across multiple queries
- Real-time loading states during AI generation

## ChromaDB Persistence

Vector database stored in `./chroma_db/` directory with persistent client configuration. Data survives application restarts and supports incremental document addition.