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

    # ========== Required fields for create_react_agent ==========
    # Agent conversation messages (required by LangGraph)
    messages: Annotated[NotRequired[List[BaseMessage]], messages_reducer]

    # Remaining steps counter (required by create_react_agent)
    remaining_steps: NotRequired[int]

    # ========== Deep Agent Fields ==========
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


def create_initial_deep_state(paper_path: str = "", preload_pdf: bool = True) -> DeepAgentState:
    """Create initial deep agent state for reference validation.

    Args:
        paper_path: Path to the PDF paper to validate
        preload_pdf: Whether to pre-load PDF content into virtual filesystem

    Returns:
        Initialized DeepAgentState
    """
    files = {}
    paper_content = ""
    paper_title = ""

    # Pre-load PDF content if requested and file exists
    if preload_pdf and paper_path and paper_path.endswith('.pdf'):
        expanded_path = os.path.expanduser(paper_path)
        if os.path.exists(expanded_path):
            try:
                from pypdf import PdfReader
                reader = PdfReader(expanded_path)
                num_pages = len(reader.pages)

                # Extract text from all pages
                text_parts = []
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Page {i+1} ---\n{page_text}")

                full_text = "\n\n".join(text_parts)
                char_count = len(full_text)

                # Try to extract title
                if reader.metadata and reader.metadata.title:
                    paper_title = reader.metadata.title
                elif text_parts:
                    first_lines = full_text.split('\n')[:5]
                    for line in first_lines:
                        if line.strip() and len(line.strip()) > 10:
                            paper_title = line.strip()[:100]
                            break

                paper_content = full_text

                # Save to virtual filesystem
                files["/input/paper_text.md"] = f"# {paper_title}\n\n{full_text}"
                files["/input/paper_info.md"] = f"""# Paper Information

- **Title**: {paper_title}
- **Path**: {paper_path}
- **Pages**: {num_pages}
- **Characters**: {char_count}
- **Status**: Pre-loaded into virtual filesystem
"""
                print(f"   📄 PDF pre-loaded: {paper_title[:50]}... ({num_pages} pages, {char_count:,} chars)")

            except Exception as e:
                print(f"   ⚠️ PDF pre-load failed: {e}")

    return DeepAgentState(
        # Core Deep Agent fields
        messages=[],
        todos=[],
        files=files,

        # Reference validation fields
        paper_path=paper_path,
        paper_content=paper_content,
        paper_title=paper_title,
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
