# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.


## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   # Install main dependencies
   uv sync
   
   # Install development dependencies (required for testing)
   uv sync --extra dev
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Testing

### Running Tests

All tests must pass before any new feature or change is considered completed.

```bash
# Run all tests
uv run pytest

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage report
uv run pytest --cov=backend --cov-report=term-missing

# Run specific test file
uv run pytest backend/tests/test_document_processor.py

# Run tests matching a pattern
uv run pytest -k "test_search"

# Run only unit tests (if marked)
uv run pytest -m unit

# Generate HTML coverage report
uv run pytest --cov=backend --cov-report=html:htmlcov
```

### Test Structure

The test suite is located in `backend/tests/` and includes:
- `test_document_processor.py` - Document parsing and processing tests
- `test_vector_store.py` - ChromaDB integration and search tests  
- `test_session_manager.py` - Session management tests
- `test_ai_generator.py` - AI integration tests
- `test_search_tools.py` - Search tool functionality tests
- `conftest.py` - Shared test fixtures and configuration
