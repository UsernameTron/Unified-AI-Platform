# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development
- **Start development server**: `./start-dev.sh` - Starts Flask app in development mode with live reload
- **Run tests**: `python3 test_enhanced_integration.py` - Main integration test suite
- **Run specific tests**: `python3 test_unified_system.py` - System validation tests
- **Health check**: `curl http://localhost:5000/health` - Check service health

### Docker Management
- **Build containers**: `./docker-manage.sh build [production|development]`
- **Start services**: `./docker-manage.sh start [production|development]`
- **View logs**: `./docker-manage.sh logs [service] [-f]`
- **Stop services**: `./docker-manage.sh stop`
- **Service status**: `./docker-manage.sh status`

### Code Quality
- **Format code**: `black .` - Python code formatting
- **Sort imports**: `isort .` - Import organization
- **Lint**: `flake8 .` - Code linting
- **Type check**: `pylint .` - Static analysis

### Testing
- **Run pytest**: `pytest shared_agents/tests/` - Unit tests for shared agents
- **Coverage**: `pytest --cov=shared_agents` - Test coverage report
- **Flask tests**: `python3 test_enhanced_flask_integration.py` - API endpoint tests

## Architecture Overview

This is a hybrid AI system that combines VectorDBRAG and MindMeld-v1.1 capabilities while maintaining separation between the original projects.

### Core Components
- **`shared_agents/`** - Extracted MindMeld agent framework providing unified agent interface
- **`rag_integration/`** - Enhanced Flask application with RAG capabilities
- **`agent_system/`** - Web interface and analytics dashboard
- **`VectorDBRAG/`** (external) - Vector database and RAG functionality
- **`MindMeld-v1.1/`** (external) - Original MindMeld project (unchanged)

### Agent System
The system provides 10 enhanced agent types through `shared_agents/core/agent_factory.py`:
- CEO, Research, Triage, Code Analysis, Code Debugger, Code Repair, Performance Profiler, Test Generator, Image, Audio

Each agent supports capability-based discovery via `AgentCapability` enum covering:
- Code analysis, debugging, repair, performance analysis, test generation
- Speech/audio analysis, visual analysis, strategic planning, research
- Vector search, RAG processing

### Configuration
- **Environment variables**: Set `OPENAI_API_KEY`, optionally `OLLAMA_HOST` and `LOCAL_MODEL`
- **Multi-model support**: OpenAI GPT models and local Ollama models
- **Config validation**: Automatic validation on startup via `config/shared_config.py`

### API Endpoints
- **Enhanced agents**: `POST /api/enhanced/agents/query` - Query specific agent types
- **Health check**: `GET /health` - Service health status
- **Analytics**: Various endpoints for system monitoring

## Development Setup

### Prerequisites
```bash
export OPENAI_API_KEY="your-api-key"
# Optional for local models:
export OLLAMA_HOST="http://localhost:11434"
export LOCAL_MODEL="phi3.5"
```

### Installation
```bash
pip install -r requirements-dev.txt  # Development dependencies
pip install -r rag_integration/requirements.txt  # Core dependencies
```

### File Structure
- Python path includes: `/app:/app/VectorDBRAG:/app/agent_system:/app/shared_agents`
- Data directories: `/app/data/chromadb`, `/app/data/sessions`
- Logs: `/app/logs`

## Testing Strategy

### Test Files
- `test_enhanced_integration.py` - Main integration tests
- `test_unified_system.py` - System validation
- `shared_agents/tests/test_enhanced_agents.py` - Unit tests
- `test_enhanced_flask_integration.py` - API tests

### Validation Scripts
- `final_validation.py` - Comprehensive system validation
- `simple_validate.py` - Quick validation checks
- `validate_end_to_end.py` - End-to-end testing

## Important Notes

- This is a defensive AI system for security analysis and agent orchestration
- The hybrid architecture maintains clean separation between VectorDBRAG and MindMeld projects
- All agents inherit from `AgentBase` class in `shared_agents/core/agent_factory.py`
- Configuration validation happens at startup - missing API keys will prevent startup
- Development mode enables Flask debug mode, Jupyter notebooks, and live reload
- Docker deployment supports both production and development configurations