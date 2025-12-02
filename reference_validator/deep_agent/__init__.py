"""
Deep Agent Architecture for Reference Validation System

Based on teddynote-lab/deep-agents-from-scratch architecture, implementing:
- Context Offloading: Virtual file system for storing research results
- Sub-agent Delegation: Specialized agents for different tasks
- Strategic Thinking: Think tool for reflection and planning
- TODO Management: Task tracking for complex workflows

Components:
- state: DeepAgentState with TODO tracking and virtual file system
- prompts: System prompts and tool descriptions
- file_tools: Virtual file system tools (ls, read_file, write_file)
- todo_tools: TODO management tools (write_todos, read_todos)
- research_tools: Search and think tools
- task_tool: Sub-agent delegation tool
"""

from .state import (
    DeepAgentState,
    Todo,
    file_reducer,
)

from .file_tools import (
    ls,
    read_file,
    write_file,
)

from .todo_tools import (
    write_todos,
    read_todos,
)

from .research_tools import (
    think_tool,
    create_tavily_search_tool,
)

from .pdf_tools import (
    read_pdf,
    preload_pdf_to_filesystem,
    extract_text_from_pdf,
    chunk_text,
)

from .task_tool import (
    SubAgent,
    create_task_tool,
    get_default_subagents,
)

from .deep_supervisor import (
    DeepAgentSupervisor,
    create_deep_agent_supervisor,
)

from .prompts import (
    WRITE_TODOS_DESCRIPTION,
    TODO_USAGE_INSTRUCTIONS,
    LS_DESCRIPTION,
    READ_FILE_DESCRIPTION,
    WRITE_FILE_DESCRIPTION,
    FILE_USAGE_INSTRUCTIONS,
    SUMMARIZE_WEB_SEARCH,
    RESEARCHER_INSTRUCTIONS,
    VALIDATOR_INSTRUCTIONS,
    SUPERVISOR_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
)

__all__ = [
    # State
    "DeepAgentState",
    "Todo",
    "file_reducer",
    # File tools
    "ls",
    "read_file",
    "write_file",
    # PDF tools
    "read_pdf",
    "preload_pdf_to_filesystem",
    "extract_text_from_pdf",
    "chunk_text",
    # TODO tools
    "write_todos",
    "read_todos",
    # Research tools
    "think_tool",
    "create_tavily_search_tool",
    # Task tool
    "SubAgent",
    "create_task_tool",
    "get_default_subagents",
    # Deep Supervisor
    "DeepAgentSupervisor",
    "create_deep_agent_supervisor",
    # Prompts
    "WRITE_TODOS_DESCRIPTION",
    "TODO_USAGE_INSTRUCTIONS",
    "LS_DESCRIPTION",
    "READ_FILE_DESCRIPTION",
    "WRITE_FILE_DESCRIPTION",
    "FILE_USAGE_INSTRUCTIONS",
    "SUMMARIZE_WEB_SEARCH",
    "RESEARCHER_INSTRUCTIONS",
    "VALIDATOR_INSTRUCTIONS",
    "SUPERVISOR_INSTRUCTIONS",
    "SUBAGENT_USAGE_INSTRUCTIONS",
]
