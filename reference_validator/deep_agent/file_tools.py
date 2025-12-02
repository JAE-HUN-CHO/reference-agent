"""
Virtual file system tools for Deep Agent state management.

Based on teddynote-lab/deep-agents-from-scratch file_tools.py

This module provides tools for managing a virtual filesystem stored in agent state,
enabling context offloading and information persistence across agent interactions.

Also includes tools for accessing real filesystem and PDF parsing.
"""

import os
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .prompts import (
    LS_DESCRIPTION,
    READ_FILE_DESCRIPTION,
    WRITE_FILE_DESCRIPTION,
    READ_REAL_FILE_DESCRIPTION,
    PARSE_PDF_DESCRIPTION,
)
from .state import DeepAgentState


@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files in the virtual filesystem.
    
    Returns:
        List of file paths in the virtual filesystem
    """
    files = state.get("files", {})
    if not files:
        return ["(empty - no files in virtual filesystem)"]
    return sorted(list(files.keys()))


@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content from virtual filesystem with optional offset and limit.

    Args:
        file_path: Path to the file to read
        state: Agent state containing virtual filesystem (injected in tool node)
        offset: Line number to start reading from (default: 0)
        limit: Maximum number of lines to read (default: 2000)

    Returns:
        Formatted file content with line numbers, or error message if file not found
    """
    files = state.get("files", {})
    
    if file_path not in files:
        # Try with different path variations
        for key in files.keys():
            if key.endswith(file_path) or file_path.endswith(key):
                file_path = key
                break
        else:
            return f"Error: File '{file_path}' not found. Available files: {list(files.keys())}"

    content = files[file_path]
    if not content:
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]  # Truncate long lines
        result_lines.append(f"{i + 1:6d}\t{line_content}")

    return "\n".join(result_lines)


@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write content to a file in the virtual filesystem.

    Args:
        file_path: Path where the file should be created/updated
        content: Content to write to the file
        state: Agent state containing virtual filesystem (injected in tool node)
        tool_call_id: Tool call identifier for message response (injected in tool node)

    Returns:
        Command to update agent state with new file content
    """
    files = state.get("files", {})
    files[file_path] = content
    
    # Calculate file size info
    line_count = len(content.splitlines())
    char_count = len(content)
    
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(
                    f"Successfully wrote to {file_path} ({line_count} lines, {char_count} chars)",
                    tool_call_id=tool_call_id
                )
            ],
        }
    )


def save_to_file(state: DeepAgentState, file_path: str, content: str) -> DeepAgentState:
    """Helper function to save content to virtual file system.
    
    Args:
        state: Current agent state
        file_path: Path to save the file
        content: Content to write
        
    Returns:
        Updated state with new file
    """
    files = state.get("files", {})
    files[file_path] = content
    state["files"] = files
    return state


def read_from_file(state: DeepAgentState, file_path: str) -> str:
    """Helper function to read content from virtual file system.
    
    Args:
        state: Current agent state
        file_path: Path to read from
        
    Returns:
        File content or error message
    """
    files = state.get("files", {})
    return files.get(file_path, f"Error: File '{file_path}' not found")


def list_files(state: DeepAgentState) -> list[str]:
    """Helper function to list all files in virtual file system.

    Args:
        state: Current agent state

    Returns:
        List of file paths
    """
    return list(state.get("files", {}).keys())


# ============================================================================
# Real Filesystem Tools
# ============================================================================

@tool(description=READ_REAL_FILE_DESCRIPTION, parse_docstring=True)
def read_real_file(
    file_path: str,
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read content from a real file on the actual filesystem.

    Args:
        file_path: Path to the real file on disk
        offset: Line number to start reading from (default: 0)
        limit: Maximum number of lines to read (default: 2000)

    Returns:
        Formatted file content with line numbers, or error message if file not found
    """
    # Expand user home directory and resolve path
    expanded_path = os.path.expanduser(file_path)

    if not os.path.exists(expanded_path):
        return f"Error: File '{file_path}' not found on disk."

    if not os.path.isfile(expanded_path):
        return f"Error: '{file_path}' is not a file (might be a directory)."

    # Check if it's a binary file (like PDF)
    if file_path.lower().endswith('.pdf'):
        return f"Error: '{file_path}' is a PDF file. Use parse_pdf tool instead."

    try:
        with open(expanded_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = []
    for i in range(start_idx, end_idx):
        line_content = lines[i][:2000]  # Truncate long lines
        result_lines.append(f"{i + 1:6d}\t{line_content}")

    file_info = f"[Real file: {file_path} | Total lines: {len(lines)} | Showing: {start_idx+1}-{end_idx}]\n"
    return file_info + "\n".join(result_lines)


@tool(description=PARSE_PDF_DESCRIPTION, parse_docstring=True)
def parse_pdf(
    pdf_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Extract text content from a PDF file and save to virtual filesystem.

    Args:
        pdf_path: Path to the PDF file to parse
        state: Agent state containing virtual filesystem (injected in tool node)
        tool_call_id: Tool call identifier for message response (injected in tool node)

    Returns:
        Command to update agent state with extracted PDF content
    """
    # Expand user home directory and resolve path
    expanded_path = os.path.expanduser(pdf_path)

    if not os.path.exists(expanded_path):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: PDF file '{pdf_path}' not found on disk.",
                        tool_call_id=tool_call_id
                    )
                ],
            }
        )

    try:
        # Import pypdf for PDF parsing
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

        # Try to extract title (usually first line or metadata)
        title = "Unknown"
        if reader.metadata and reader.metadata.title:
            title = reader.metadata.title
        elif text_parts:
            # Try first non-empty line
            first_lines = full_text.split('\n')[:5]
            for line in first_lines:
                if line.strip() and len(line.strip()) > 10:
                    title = line.strip()[:100]
                    break

        # Save to virtual filesystem
        files = state.get("files", {})
        files["/input/paper_text.md"] = f"# {title}\n\n{full_text}"
        files["/input/paper_info.md"] = f"""# Paper Information

- **Title**: {title}
- **Path**: {pdf_path}
- **Pages**: {num_pages}
- **Characters**: {char_count}
- **Extracted**: Successfully
"""

        result_message = f"""PDF parsed successfully!

📄 **Title**: {title}
📑 **Pages**: {num_pages}
📝 **Characters**: {char_count:,}

Content saved to:
- /input/paper_text.md (full text)
- /input/paper_info.md (metadata)

Use read_file('/input/paper_text.md') to read the extracted content."""

        return Command(
            update={
                "files": files,
                "messages": [
                    ToolMessage(result_message, tool_call_id=tool_call_id)
                ],
            }
        )

    except ImportError:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        "Error: pypdf library not installed. Run 'pip install pypdf'",
                        tool_call_id=tool_call_id
                    )
                ],
            }
        )
    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error parsing PDF: {str(e)}",
                        tool_call_id=tool_call_id
                    )
                ],
            }
        )


if __name__ == "__main__":
    # Test
    print("=== File Tools 테스트 ===")
    
    from state import create_initial_deep_state
    
    # Create test state
    state = create_initial_deep_state("test.pdf")
    
    # Test save_to_file
    state = save_to_file(state, "test.md", "# Test File\n\nThis is test content.\n")
    print(f"파일 저장 후: {list_files(state)}")
    
    # Test read_from_file
    content = read_from_file(state, "test.md")
    print(f"\n파일 내용:\n{content}")
    
    # Test non-existent file
    error = read_from_file(state, "nonexistent.md")
    print(f"\n존재하지 않는 파일: {error}")
