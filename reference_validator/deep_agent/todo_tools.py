"""
TODO management tools for task planning and progress tracking.

Based on teddynote-lab/deep-agents-from-scratch todo_tools.py

This module provides tools for creating and managing structured task lists
that enable agents to plan complex workflows and track progress through
multi-step reference validation operations.
"""

from typing import Annotated, List

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .prompts import WRITE_TODOS_DESCRIPTION
from .state import DeepAgentState, Todo


@tool(description=WRITE_TODOS_DESCRIPTION, parse_docstring=True)
def write_todos(
    todos: List[Todo],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create or update the agent's TODO list for task planning and tracking.

    Args:
        todos: List of Todo items with content and status
        tool_call_id: Tool call identifier for message response

    Returns:
        Command to update agent state with new TODO list
    """
    # Format todos for display
    formatted = []
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo["status"], "❓")
        formatted.append(f"{i}. {emoji} {todo['content']} ({todo['status']})")
    
    todo_display = "\n".join(formatted) if formatted else "No todos"
    
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(
                    f"Updated TODO list:\n{todo_display}",
                    tool_call_id=tool_call_id
                )
            ],
        }
    )


@tool(parse_docstring=True)
def read_todos(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read the current TODO list from the agent state.

    This tool allows the agent to retrieve and review the current TODO list
    to stay focused on remaining tasks and track progress through complex workflows.

    Args:
        state: Injected agent state containing the current TODO list
        tool_call_id: Injected tool call identifier for message tracking

    Returns:
        Command to update agent state with ToolMessage containing formatted TODO list
    """
    todos = state.get("todos", [])
    
    if not todos:
        message_content = "📋 No todos currently in the list."
    else:
        result_lines = ["📋 Current TODO List:"]
        result_lines.append("-" * 40)
        
        # Group by status
        pending = []
        in_progress = []
        completed = []
        
        for i, todo in enumerate(todos, 1):
            status = todo.get("status", "pending")
            content = todo.get("content", "")
            
            if status == "completed":
                completed.append(f"  ✅ {content}")
            elif status == "in_progress":
                in_progress.append(f"  🔄 {content}")
            else:
                pending.append(f"  ⏳ {content}")
        
        if in_progress:
            result_lines.append("\n🔄 IN PROGRESS:")
            result_lines.extend(in_progress)
        
        if pending:
            result_lines.append("\n⏳ PENDING:")
            result_lines.extend(pending)
        
        if completed:
            result_lines.append("\n✅ COMPLETED:")
            result_lines.extend(completed)
        
        # Summary
        result_lines.append("-" * 40)
        result_lines.append(f"Total: {len(todos)} | In Progress: {len(in_progress)} | Pending: {len(pending)} | Completed: {len(completed)}")
        
        message_content = "\n".join(result_lines)

    return Command(
        update={
            "messages": [
                ToolMessage(message_content, tool_call_id=tool_call_id)
            ],
        }
    )


# Helper functions for programmatic TODO management

def create_todos(tasks: List[str]) -> List[Todo]:
    """Create a list of pending TODO items from task descriptions.
    
    Args:
        tasks: List of task descriptions
        
    Returns:
        List of Todo items with pending status
    """
    return [
        Todo(content=task, status="pending")
        for task in tasks
    ]


def update_todo_status(
    todos: List[Todo],
    task_content: str,
    new_status: str
) -> List[Todo]:
    """Update the status of a specific TODO item.
    
    Args:
        todos: Current TODO list
        task_content: Content of the task to update (partial match)
        new_status: New status to set
        
    Returns:
        Updated TODO list
    """
    updated_todos = []
    for todo in todos:
        if task_content.lower() in todo["content"].lower():
            updated_todos.append(Todo(
                content=todo["content"],
                status=new_status
            ))
        else:
            updated_todos.append(todo)
    return updated_todos


def get_next_pending_todo(todos: List[Todo]) -> Todo | None:
    """Get the next pending TODO item.
    
    Args:
        todos: Current TODO list
        
    Returns:
        First pending Todo or None if no pending tasks
    """
    for todo in todos:
        if todo.get("status") == "pending":
            return todo
    return None


def get_in_progress_todo(todos: List[Todo]) -> Todo | None:
    """Get the currently in-progress TODO item.
    
    Args:
        todos: Current TODO list
        
    Returns:
        In-progress Todo or None if no task is in progress
    """
    for todo in todos:
        if todo.get("status") == "in_progress":
            return todo
    return None


def format_todos_for_display(todos: List[Todo]) -> str:
    """Format TODO list for display.
    
    Args:
        todos: List of Todo items
        
    Returns:
        Formatted string representation
    """
    if not todos:
        return "No todos"
    
    lines = []
    for i, todo in enumerate(todos, 1):
        status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
        emoji = status_emoji.get(todo.get("status", "pending"), "❓")
        lines.append(f"{i}. {emoji} {todo.get('content', '')} ({todo.get('status', 'pending')})")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test
    print("=== TODO Tools 테스트 ===")
    
    # Create todos
    todos = create_todos([
        "Parse PDF and extract references",
        "Search for reference [1]",
        "Validate reference [1]",
        "Generate final report",
    ])
    print("생성된 TODO:")
    print(format_todos_for_display(todos))
    
    # Update status
    print("\n[첫 번째 작업 시작]")
    todos = update_todo_status(todos, "Parse PDF", "in_progress")
    print(format_todos_for_display(todos))
    
    # Complete first task
    print("\n[첫 번째 작업 완료]")
    todos = update_todo_status(todos, "Parse PDF", "completed")
    print(format_todos_for_display(todos))
    
    # Get next pending
    print("\n다음 대기 작업:")
    next_todo = get_next_pending_todo(todos)
    if next_todo:
        print(f"  {next_todo['content']}")
