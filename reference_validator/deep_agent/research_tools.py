"""
Research Tools for Deep Agent Reference Validation.

Based on teddynote-lab/deep-agents-from-scratch research_tools.py

This module provides search and content processing utilities for the research agent,
including web search capabilities (Tavily), content summarization, and strategic thinking tools.
"""

import os
from datetime import datetime
from typing import Annotated, Literal, Optional, List, Dict, Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from .prompts import SUMMARIZE_WEB_SEARCH, get_current_time
from .state import DeepAgentState


# ============================================================================
# Helper Functions
# ============================================================================

def run_tavily_search(
    search_query: str,
    api_key: Optional[str] = None,
    max_results: int = 3,
    topic: Literal["general", "news", "academic"] = "general",
    include_raw_content: bool = True,
) -> dict:
    """Perform search using Tavily API for a single query.

    Args:
        search_query: Search query to execute
        api_key: Tavily API key (uses env var if not provided)
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        Search results dictionary
    """
    api_key = api_key or os.getenv("TAVILY_API_KEY")
    
    if not api_key:
        return {
            "query": search_query,
            "results": [],
            "error": "TAVILY_API_KEY not set"
        }
    
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        
        result = client.search(
            search_query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            search_depth="advanced",  # Better for academic searches
        )
        return result
        
    except ImportError:
        return {
            "query": search_query,
            "results": [],
            "error": "tavily-python not installed. Run: pip install tavily-python"
        }
    except Exception as e:
        return {
            "query": search_query,
            "results": [],
            "error": str(e)
        }


class SearchSummary(BaseModel):
    """Schema for search result summarization."""
    filename: str = Field(description="Descriptive filename for the result")
    summary: str = Field(description="Brief summary of the search result")


def summarize_search_results(
    results: List[Dict[str, Any]],
    summarization_llm=None
) -> List[SearchSummary]:
    """Summarize search results using LLM.
    
    Args:
        results: List of search result dictionaries
        summarization_llm: LLM to use for summarization (optional)
        
    Returns:
        List of SearchSummary objects
    """
    if not results:
        return []
    
    summaries = []
    
    for i, result in enumerate(results):
        title = result.get("title", "Unknown")
        url = result.get("url", "")
        content = result.get("raw_content", result.get("content", result.get("snippet", "")))
        
        # Create simple summary without LLM call
        summary_text = content[:500] if content else f"Result for: {title}"
        
        # Generate filename
        safe_title = "".join(c for c in title[:30] if c.isalnum() or c in "_ -").strip()
        safe_title = safe_title.replace(" ", "_").lower()
        filename = f"search_{safe_title}_{i}.md" if safe_title else f"search_result_{i}.md"
        
        summaries.append(SearchSummary(
            filename=filename,
            summary=summary_text
        ))
    
    return summaries


def process_search_results(results: dict) -> List[Dict[str, Any]]:
    """Process raw search results into structured format.
    
    Args:
        results: Raw Tavily search results
        
    Returns:
        List of processed result dictionaries
    """
    search_results = results.get("results", [])
    
    if not search_results:
        return []
    
    processed = []
    for result in search_results:
        processed.append({
            "url": result.get("url", ""),
            "title": result.get("title", ""),
            "content": result.get("raw_content", result.get("content", "")),
            "snippet": result.get("content", ""),
            "score": result.get("score", 0.0),
        })
    
    return processed


# ============================================================================
# Tools
# ============================================================================

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to validate the reference?
    - When assessing validation gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a confident validation now?
    - How complex is the reference: Have I reached the search limit?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence for validation?
    4. Strategic decision - Should I continue searching or provide my validation?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"💭 Reflection recorded:\n{reflection}\n\nNow proceed with your next action based on this reflection."


def create_tavily_search_tool(
    api_key: Optional[str] = None,
    max_results: int = 3,
):
    """Create a Tavily search tool with configured parameters.
    
    Args:
        api_key: Tavily API key
        max_results: Maximum results per search
        
    Returns:
        Configured tavily_search tool
    """
    
    @tool(parse_docstring=True)
    def tavily_search(
        query: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Perform web search and save detailed results to files while returning minimal context.

        Performs web search and saves full content to files for context offloading.
        Returns only essential information to help the agent decide next steps.

        Args:
            query: Search query to execute
            state: Agent state for file storage (injected)
            tool_call_id: Tool call identifier (injected)

        Returns:
            Command with file updates and summary message
        """
        # Perform search
        search_results = run_tavily_search(
            query,
            api_key=api_key or os.getenv("TAVILY_API_KEY"),
            max_results=max_results,
            include_raw_content=True,
        )
        
        # Check for errors
        if "error" in search_results:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"⚠️ Search error: {search_results['error']}\nQuery: {query}",
                            tool_call_id=tool_call_id
                        )
                    ],
                }
            )
        
        # Process results
        processed_results = process_search_results(search_results)
        summaries = summarize_search_results(processed_results)
        
        # Save each result to virtual file system
        files = state.get("files", {})
        saved_files = []
        summary_lines = []
        
        for i, (result, summary) in enumerate(zip(processed_results, summaries)):
            filename = f"search/{summary.filename}"
            
            # Create file content
            file_content = f"""# Search Result: {result['title']}

**URL:** {result['url']}
**Query:** {query}
**Date:** {get_current_time()}
**Score:** {result.get('score', 'N/A')}

## Summary
{summary.summary}

## Content
{result.get('content', 'No content available')[:3000]}
"""
            
            files[filename] = file_content
            saved_files.append(filename)
            summary_lines.append(f"  - {filename}: {summary.summary[:100]}...")
        
        # Create summary message
        if processed_results:
            summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summary_lines)}

📁 Files saved: {', '.join(saved_files)}
💡 Use read_file() to access full details when needed."""
        else:
            summary_text = f"🔍 No results found for '{query}'. Try different search terms."
        
        return Command(
            update={
                "files": files,
                "messages": [ToolMessage(summary_text, tool_call_id=tool_call_id)],
            }
        )
    
    return tavily_search


# Create default tavily_search tool
tavily_search = create_tavily_search_tool()


if __name__ == "__main__":
    print("=== Research Tools 테스트 ===")
    
    # Test think_tool
    print("\n1. Think Tool 테스트:")
    result = think_tool.invoke({"reflection": "I found 2 results for the reference search. The titles match well but I need to verify the authors."})
    print(result)
    
    # Test Tavily search (requires API key)
    print("\n2. Tavily Search 테스트:")
    api_key = os.getenv("TAVILY_API_KEY")
    if api_key:
        results = run_tavily_search("Attention is all you need Vaswani 2017", api_key=api_key)
        if "error" not in results:
            print(f"   결과 수: {len(results.get('results', []))}")
            for r in results.get("results", [])[:2]:
                print(f"   - {r.get('title', 'N/A')[:50]}...")
        else:
            print(f"   오류: {results['error']}")
    else:
        print("   TAVILY_API_KEY not set, skipping test")
