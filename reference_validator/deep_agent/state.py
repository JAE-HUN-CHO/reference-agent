"""
Deep Agent State Management

Based on teddynote-lab/deep-agents-from-scratch state.py

This module defines the extended agent state structure that supports:
- Task planning and progress tracking through TODO lists
- Context offloading through a virtual file system stored in state
- Reference validation specific fields
- Efficient state merging with reducer functions
"""

from typing import Annotated, Literal, NotRequired, Optional, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Todo(TypedDict):
    """A structured task item for tracking progress through complex workflows.

    Attributes:
        content: Short, specific description of the task
        status: Current state - pending, in_progress, or completed
    """
    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left: Optional[Dict[str, str]], right: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Merge two file dictionaries, with right side taking precedence.

    Used as a reducer function for the files field in agent state,
    allowing incremental updates to the virtual file system.

    Args:
        left: Left side dictionary (existing files)
        right: Right side dictionary (new/updated files)

    Returns:
        Merged dictionary with right values overriding left values
    """
    if left is None:
        return right or {}
    elif right is None:
        return left or {}
    else:
        return {**left, **right}


def messages_reducer(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    """Merge message lists by appending right to left.
    
    Args:
        left: Existing messages
        right: New messages to append
        
    Returns:
        Combined message list
    """
    if left is None:
        return right or []
    elif right is None:
        return left or []
    else:
        return left + right


class DeepAgentState(TypedDict):
    """Extended agent state that includes task tracking and virtual file system.

    This state extends the basic LangGraph state pattern with:
    - todos: List of Todo items for task planning and progress tracking
    - files: Virtual file system stored as dict mapping filenames to content
    - messages: Agent conversation history
    - Reference validation specific fields
    """
    
    # ========== Core Deep Agent Fields ==========
    # Agent conversation messages
    messages: Annotated[NotRequired[List[BaseMessage]], messages_reducer]
    
    # Task planning and progress tracking
    todos: NotRequired[List[Todo]]
    
    # Virtual file system (filename -> content mapping)
    # Uses file_reducer for merging updates
    files: Annotated[NotRequired[Dict[str, str]], file_reducer]
    
    # ========== Reference Validation Fields ==========
    # Input
    paper_path: NotRequired[str]
    paper_content: NotRequired[str]
    paper_title: NotRequired[str]
    
    # Parsed references
    references: NotRequired[List[Dict[str, Any]]]
    citation_contexts: NotRequired[List[Dict[str, Any]]]
    
    # Current processing state
    current_ref_index: NotRequired[int]
    current_reference: NotRequired[Optional[Dict[str, Any]]]
    
    # Results
    validation_results: NotRequired[List[Dict[str, Any]]]
    final_report: NotRequired[Optional[Dict[str, Any]]]
    
    # Metadata
    error_log: NotRequired[List[str]]
    processing_log: NotRequired[List[str]]


def create_initial_deep_state(paper_path: str = "") -> DeepAgentState:
    """Create initial deep agent state for reference validation.
    
    Args:
        paper_path: Path to the PDF paper to validate
        
    Returns:
        Initialized DeepAgentState
    """
    return DeepAgentState(
        # Core Deep Agent fields
        messages=[],
        todos=[],
        files={},
        
        # Reference validation fields
        paper_path=paper_path,
        paper_content="",
        paper_title="",
        references=[],
        citation_contexts=[],
        current_ref_index=0,
        current_reference=None,
        validation_results=[],
        final_report=None,
        error_log=[],
        processing_log=[],
    )


if __name__ == "__main__":
    # Test
    print("=== Deep Agent State 테스트 ===")
    
    # Create initial state
    state = create_initial_deep_state("test.pdf")
    print(f"\n초기 상태 생성 완료")
    print(f"파일: {state.get('paper_path')}")
    print(f"TODO 수: {len(state.get('todos', []))}")
    print(f"파일 시스템: {state.get('files', {})}")
    
    # Test file reducer
    print("\n=== file_reducer 테스트 ===")
    files1 = {"a.txt": "content a", "b.txt": "content b"}
    files2 = {"b.txt": "new content b", "c.txt": "content c"}
    merged = file_reducer(files1, files2)
    print(f"병합 결과: {merged}")
    
    # Test TODO
    print("\n=== TODO 테스트 ===")
    todo: Todo = {"content": "Test task", "status": "pending"}
    print(f"TODO: {todo}")
