"""
PDF reading tools for Deep Agent with context-aware chunking.

This module provides tools for reading PDF files and loading them into the virtual
file system in manageable chunks to avoid exceeding context limits.
"""

import os
from typing import Annotated, Optional, Tuple, List, Dict

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from .state import DeepAgentState


def extract_text_from_pdf(pdf_path: str) -> Tuple[str, str, int]:
    """
    Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple of (full_text, title, page_count)
    """
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    text_parts = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text_parts.append(f"--- Page {i + 1} ---\n{page_text}")

    full_text = "\n\n".join(text_parts)

    # Extract title from first page
    title = _extract_title(text_parts[0] if text_parts else "")

    return full_text, title, len(reader.pages)


def _extract_title(first_page_text: str) -> str:
    """Extract paper title from first page text."""
    lines = first_page_text.split('\n')

    candidate_lines = []
    for line in lines[:15]:
        line = line.strip()
        # Skip page marker
        if line.startswith("--- Page"):
            continue
        if len(line) > 10 and not line.startswith('http'):
            candidate_lines.append(line)

    if candidate_lines:
        title = max(candidate_lines, key=len)
        return title[:200]

    return "Unknown Title"


def chunk_text(text: str, max_chars: int = 15000, overlap: int = 500) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of (chunk_text, start_char, end_char) tuples
    """
    if len(text) <= max_chars:
        return [(text, 0, len(text))]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # Try to break at paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', start + max_chars // 2, end)
            if para_break > start:
                end = para_break
            else:
                # Look for sentence break
                sent_break = text.rfind('. ', start + max_chars // 2, end)
                if sent_break > start:
                    end = sent_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, min(end, len(text))))

        start = end - overlap if end < len(text) else len(text)

    return chunks


def find_references_section(text: str) -> Tuple[str, str, int]:
    """
    Split paper into body and references section.

    Args:
        text: Full paper text

    Returns:
        Tuple of (body_text, references_text, ref_start_char)
    """
    import re

    patterns = [
        r'\n\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*\n',
        r'\n\s*\d+\.\s*(References|REFERENCES)\s*\n',
        r'\n\s*References\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            return (
                text[:match.start()].strip(),
                text[match.end():].strip(),
                match.start()
            )

    # Fallback: assume last 20% is references
    split_point = int(len(text) * 0.8)
    return text[:split_point].strip(), text[split_point:].strip(), split_point


def preload_pdf_to_filesystem(
    pdf_path: str,
    max_chunk_chars: int = 15000,
) -> Tuple[Dict[str, str], str, int, int]:
    """
    Pre-load PDF content into virtual file system as chunks.

    Args:
        pdf_path: Path to the PDF file
        max_chunk_chars: Maximum characters per chunk

    Returns:
        Tuple of:
        - files: Dictionary of file_path -> content for virtual filesystem
        - title: Extracted paper title
        - page_count: Number of pages
        - total_chars: Total number of characters in full text
    """
    files = {}

    # Extract full text
    full_text, title, page_count = extract_text_from_pdf(pdf_path)
    total_chars = len(full_text)

    # Create metadata file
    files["/input/paper_info.md"] = f"""# Paper Metadata

**Title:** {title}
**Path:** {pdf_path}
**Pages:** {page_count}
**Total Characters:** {total_chars:,}

## File Structure
- `/input/paper_info.md` - This file (metadata)
- `/input/body_chunk_*.md` - Paper body content (chunked)
- `/input/references.md` - References section
"""

    # Split into body and references
    body_text, ref_text, _ = find_references_section(full_text)

    # Save references section (usually fits in one chunk)
    if len(ref_text) > max_chunk_chars:
        ref_chunks = chunk_text(ref_text, max_chunk_chars)
        for i, (chunk, start, end) in enumerate(ref_chunks):
            files[f"/input/references_part_{i+1}.md"] = f"""# References (Part {i+1}/{len(ref_chunks)})

Characters {start:,} - {end:,}

---

{chunk}
"""
    else:
        files["/input/references.md"] = f"""# References Section

{ref_text}
"""

    # Chunk body text
    body_chunks = chunk_text(body_text, max_chunk_chars)

    for i, (chunk, start, end) in enumerate(body_chunks):
        files[f"/input/body_chunk_{i+1}.md"] = f"""# Paper Body (Chunk {i+1}/{len(body_chunks)})

Characters {start:,} - {end:,}

---

{chunk}
"""

    # Update metadata with chunk info
    chunk_list = "\n".join([f"- `/input/body_chunk_{i+1}.md`" for i in range(len(body_chunks))])
    files["/input/paper_info.md"] += f"\n## Body Chunks\n{chunk_list}\n"

    return files, title, page_count, total_chars


@tool(parse_docstring=True)
def read_pdf(
    pdf_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read a PDF file and load its content into the virtual file system.

    This tool extracts text from a PDF and chunks it into manageable parts stored
    in the virtual file system. Use ls() and read_file() to access the content.

    Args:
        pdf_path: Path to the PDF file to read
        state: Agent state for file storage (injected)
        tool_call_id: Tool call identifier (injected)

    Returns:
        Command with file updates and summary message
    """
    if not os.path.exists(pdf_path):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: PDF file not found: {pdf_path}",
                        tool_call_id=tool_call_id
                    )
                ],
            }
        )

    try:
        files, title, page_count, total_chars = preload_pdf_to_filesystem(pdf_path)

        # Merge with existing files
        existing_files = state.get("files", {})
        existing_files.update(files)

        summary = f"""📄 PDF loaded successfully!

**Title:** {title}
**Pages:** {page_count}
**Characters:** {total_chars:,}

**Files created:**
{chr(10).join([f"  - {path}" for path in sorted(files.keys())])}

💡 Use `ls()` to see all files, then `read_file(path)` to read specific content.
📋 Start with `/input/paper_info.md` for an overview, then read `/input/references.md` for the reference list.
"""

        return Command(
            update={
                "files": existing_files,
                "messages": [ToolMessage(summary, tool_call_id=tool_call_id)],
            }
        )

    except Exception as e:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error reading PDF: {str(e)}",
                        tool_call_id=tool_call_id
                    )
                ],
            }
        )


if __name__ == "__main__":
    print("=== PDF Tools Test ===")

    # Test chunking
    test_text = "A" * 50000
    chunks = chunk_text(test_text, max_chars=15000)
    print(f"\nChunking test: {len(test_text)} chars -> {len(chunks)} chunks")
    for i, (chunk, start, end) in enumerate(chunks):
        print(f"  Chunk {i+1}: {len(chunk)} chars ({start}-{end})")
