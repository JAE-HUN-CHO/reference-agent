# CLAUDE.md - Reference Validator Agent

## Project Overview

This is a **Multi-Agent System for Academic Paper Reference Validation** built on the DeepAgents architecture. The system validates references in academic papers by:
- Extracting references and citation contexts from PDF papers
- Verifying reference existence through web search
- Analyzing citation appropriateness and context relevance
- Generating comprehensive validation reports

## Architecture

The system supports two execution modes:

### 1. Deep Agent Mode (Default, Recommended)
Based on teddynote-lab/deep-agents-from-scratch architecture:
- **Context Offloading**: Virtual file system for storing research results
- **Sub-agent Delegation**: Parallel reference searches via sub-agents
- **Strategic Thinking**: Think tool for reflection and planning
- **TODO Management**: Task tracking for complex workflows

### 2. LangGraph Workflow Mode
Traditional pipeline with specialized agents:
```
Parser Agent -> Web Agent -> Validation Agent -> Context Agent -> Citation Agent -> Report
```

## Project Structure

```
reference_validator/
├── __init__.py              # Package initialization, exports validate_paper()
├── main.py                  # CLI entry point
├── config.py                # LLM configuration (Ollama/OpenAI/Anthropic/Google)
├── schema.py                # Pydantic data models and LangGraph state
├── supervisor.py            # LangGraph workflow orchestration
├── requirements.txt         # Dependencies
│
├── agents/                  # Specialized agents (LangGraph mode)
│   ├── parser_agent.py      # PDF parsing, reference extraction
│   ├── web_agent.py         # Web search coordination
│   ├── validation_agent.py  # Reference existence validation
│   ├── context_agent.py     # Citation context analysis
│   └── citation_agent.py    # Citation appropriateness judgment
│
├── deep_agent/              # Deep Agent architecture (default mode)
│   ├── deep_supervisor.py   # Main Deep Agent supervisor
│   ├── state.py             # DeepAgentState with TODO and virtual FS
│   ├── prompts.py           # System prompts and tool descriptions
│   ├── file_tools.py        # Virtual file system (ls, read, write)
│   ├── todo_tools.py        # TODO management tools
│   ├── research_tools.py    # Search and think tools
│   └── task_tool.py         # Sub-agent delegation
│
└── tools/
    └── search_tools.py      # Tavily, Semantic Scholar, CrossRef integrations
```

## Key Components

### Data Models (schema.py)
- `Reference`: Parsed reference information (title, authors, year, DOI)
- `CitationContext`: In-text citation context and surrounding text
- `SearchResult`: Web search results from various sources
- `ExistenceValidationResult`: Reference existence verification result
- `CitationValidationResult`: Citation appropriateness assessment
- `ValidationReport`: Final comprehensive report

### LLM Configuration (config.py)
Supports multiple providers:
- **Ollama** (default): `glm-4.6:cloud`, `llama3.2`, `qwen2.5`, `gemma2`, `mistral`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`
- **Anthropic**: `claude-sonnet-4-20250514`, `claude-3-haiku`
- **Google**: `gemini-2.5-pro`, `gemini-2.5-flash`

### Search Tools (tools/search_tools.py)
- `TavilySearchTool`: Web search (requires TAVILY_API_KEY)
- `SemanticScholarTool`: Academic paper search (free tier available)
- `CrossRefTool`: DOI-based paper lookup (free)
- `CrawlerTool`: Direct URL content extraction
- `UnifiedSearchTool`: Combines all search sources

## Development Workflows

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r reference_validator/requirements.txt

# For local LLM (Ollama)
ollama pull llama3.2
```

### Running the System
```bash
# Deep Agent mode (default)
python reference_validator/main.py paper.pdf

# LangGraph mode
python reference_validator/main.py paper.pdf --mode langgraph

# With specific LLM
python reference_validator/main.py paper.pdf --provider openai --model gpt-4o

# Limit references
python reference_validator/main.py paper.pdf --max-refs 10
```

### Python API Usage
```python
from reference_validator import validate_paper, configure

# Configure LLM
configure(provider="ollama", model_name="llama3.2")

# Run validation
report = validate_paper("paper.pdf")
print(f"Valid: {report.valid_count}/{report.total_references}")
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `TAVILY_API_KEY` | Tavily web search API | Recommended |
| `OPENAI_API_KEY` | OpenAI API key | If using OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | If using Claude |
| `GOOGLE_API_KEY` | Google AI API key | If using Gemini |

## Code Conventions

### Language
- Code comments and docstrings are in **Korean** (한국어)
- Variable names and function names follow Python conventions (snake_case)

### Type Hints
- All functions should have type hints
- Use Pydantic models for structured data
- Use TypedDict for LangGraph state

### Error Handling
- Agents include fallback mechanisms (e.g., rule-based parsing when LLM fails)
- Errors are logged to `error_log` in state
- Processing steps logged to `processing_log`

### Agent Pattern
Each agent follows this pattern:
```python
class AgentName:
    def __init__(self, llm=None):
        self.llm = llm or settings.get_<role>_llm()

    def process(self, ...):
        # Main processing logic
        pass

def agent_name_node(state: ReferenceValidationState) -> ReferenceValidationState:
    """LangGraph node function"""
    agent = AgentName()
    # Process and update state
    return state
```

## Testing

Run individual modules:
```bash
# Test config
python reference_validator/config.py

# Test schema
python reference_validator/schema.py

# Test search tools
python reference_validator/tools/search_tools.py

# Test parser agent
python reference_validator/agents/parser_agent.py
```

## Common Tasks

### Adding a New LLM Provider
1. Add provider to `LLMProvider` enum in `config.py`
2. Implement `_create_<provider>_llm()` in `LLMFactory`
3. Add preset configurations to `ModelPresets`

### Adding a New Search Source
1. Create class inheriting from `BaseSearchTool` in `tools/search_tools.py`
2. Implement `search()` and `is_available()` methods
3. Add to `UnifiedSearchTool`

### Modifying Agent Behavior
- Agent prompts are defined inline in each agent file
- Deep Agent prompts are centralized in `deep_agent/prompts.py`
- Modify prompts to change extraction/validation behavior

### Extending Validation Criteria
1. Update schemas in `schema.py` (e.g., add new fields to `CitationValidationResult`)
2. Modify validation logic in `validation_agent.py` or `citation_agent.py`
3. Update report generation in `supervisor.py`

## Output Format

Results are saved as JSON with structure:
```json
{
  "paper_title": "...",
  "statistics": {
    "total_references": 42,
    "valid_count": 38,
    "invalid_count": 2,
    "average_appropriateness_score": 82.5
  },
  "invalid_references": ["[15]", "[23]"],
  "summary": "...",
  "recommendations": ["..."]
}
```

## Notes for AI Assistants

1. **Primary language**: Korean documentation, English code
2. **Default mode**: Deep Agent (`--mode deep-agent`)
3. **LLM dependency**: System requires LLM access (local Ollama or cloud API)
4. **Search tools**: Tavily provides best results but requires API key; Semantic Scholar and CrossRef work without keys
5. **State management**: LangGraph mode uses `ReferenceValidationState`; Deep Agent mode uses `DeepAgentState`
6. **PDF handling**: Uses pypdf for text extraction; complex layouts may require preprocessing
