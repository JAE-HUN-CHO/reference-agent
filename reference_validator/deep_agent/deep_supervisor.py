"""
Deep Agent Supervisor for Reference Validation

Based on teddynote-lab/deep-agents-from-scratch architecture.

This module implements the main Deep Agent that orchestrates reference validation
using sub-agent delegation, virtual file system, and TODO management.

Key Features:
- Context Offloading: Store results in virtual file system
- Sub-agent Delegation: Delegate searches to research sub-agents
- Strategic Thinking: Use think_tool for reflection
- TODO Management: Track progress through complex validation
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from .state import DeepAgentState, create_initial_deep_state
from .file_tools import ls, read_file, write_file, save_to_file, read_real_file, parse_pdf
from .todo_tools import write_todos, read_todos, create_todos, format_todos_for_display
from .research_tools import think_tool, create_tavily_search_tool
from .task_tool import create_task_tool, SubAgent, get_default_subagents
from .pdf_tools import read_pdf, preload_pdf_to_filesystem
from .prompts import (
    format_supervisor_prompt,
    RESEARCHER_INSTRUCTIONS,
    get_current_time,
)


class DeepAgentSupervisor:
    """Deep Agent Supervisor for Reference Validation.

    Orchestrates the complete reference validation workflow using:
    - Sub-agent delegation for parallel reference searches
    - Virtual file system for context offloading
    - TODO management for progress tracking
    - Think tool for strategic reflection
    """

    def __init__(
        self,
        model: BaseChatModel,
        tavily_api_key: Optional[str] = None,
        max_concurrent_agents: int = 3,
        max_iterations: int = 3,
        verbose: bool = True,
    ):
        """Initialize the Deep Agent Supervisor.

        Args:
            model: LLM to use for the supervisor and sub-agents
            tavily_api_key: Tavily API key for web search
            max_concurrent_agents: Maximum parallel sub-agents
            max_iterations: Maximum task delegation iterations
            verbose: Whether to print detailed logs
        """
        self.model = model
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.max_concurrent_agents = max_concurrent_agents
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Create tools
        self.tools = self._create_tools()

        # Create agent
        self.agent = self._create_agent()

    def _create_tools(self) -> List[BaseTool]:
        """Create all tools for the supervisor agent."""
        # Sub-agent tools
        tavily_search = create_tavily_search_tool(api_key=self.tavily_api_key)
        sub_agent_tools = [tavily_search, think_tool]

        # Create task delegation tool
        subagents = get_default_subagents()
        task_tool = create_task_tool(
            sub_agent_tools,
            subagents,
            self.model,
            DeepAgentState,
        )

        # All tools available to supervisor
        all_tools = [
            # Virtual file system tools
            ls,
            read_file,
            write_file,
            # PDF reading tools (chunked for large PDFs)
            read_pdf,
            # Real file system tools (PDF 파싱 포함)
            read_real_file,
            parse_pdf,
            # TODO tools
            write_todos,
            read_todos,
            # Research tools
            think_tool,
            tavily_search,  # Supervisor can also search directly
            # Delegation tool
            task_tool,
        ]

        return all_tools

    def _create_agent(self):
        """Create the supervisor agent using LangGraph's create_react_agent."""
        # Create the system prompt
        system_prompt = format_supervisor_prompt(
            max_concurrent_research_units=self.max_concurrent_agents,
            max_researcher_iterations=self.max_iterations,
        )

        # Create react agent using LangGraph's prebuilt function
        # Pass DeepAgentState as state_schema to enable virtual filesystem access
        agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=system_prompt,
            state_schema=DeepAgentState,  # Use our custom state with files, todos
        )

        return agent

    def validate_paper(
        self,
        paper_path: str,
        max_references: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Validate references in a paper.

        Args:
            paper_path: Path to the PDF paper
            max_references: Maximum references to validate (None = all)

        Returns:
            Dictionary containing validation results and report
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Deep Agent Reference Validation")
            print(f"Paper: {paper_path}")
            print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")

        # Create initial state (preload_pdf=False since we do chunked loading below)
        state = create_initial_deep_state(paper_path, preload_pdf=False)

        # Pre-load PDF content into virtual file system with chunking to avoid context overflow
        if self.verbose:
            print("   📄 Pre-loading PDF into virtual file system (chunked)...")
        try:
            files, title, page_count, total_chars = preload_pdf_to_filesystem(
                paper_path,
                max_chunk_chars=12000,  # Conservative chunk size for context safety
            )
            state["files"] = files
            state["paper_title"] = title
            if self.verbose:
                print(f"      ✓ Loaded: {title[:50]}... ({page_count} pages, {total_chars:,} chars)")
                print(f"      ✓ Created {len(files)} files in virtual filesystem")
        except Exception as e:
            if self.verbose:
                print(f"      ⚠️ PDF pre-load warning: {e}")
            # Continue without pre-loading - agent can use parse_pdf tool

        # Build the task description (no PDF content included - it's in virtual filesystem)
        if max_references:
            task = f"""Validate references in the paper.

The PDF content has been pre-loaded into the virtual file system.

**IMPORTANT: Do NOT try to read the PDF file directly. Use the virtual file system instead.**

Your workflow:
1. Use `ls()` to see available files
2. Read `/input/paper_info.md` for paper metadata
3. Read `/input/references.md` (or references_part_*.md) to get the reference list
4. Create a TODO list to track progress
5. For each reference (up to {max_references}), search to verify it exists
6. Validate each reference against search results
7. Save results to `/validation/` directory
8. Generate a final report at `/reports/validation_report.md`

Start by using `ls()` to see the pre-loaded files."""
        else:
            task = f"""Validate ALL references in the paper.

The PDF content has been pre-loaded into the virtual file system.

**IMPORTANT: Do NOT try to read the PDF file directly. Use the virtual file system instead.**

Your workflow:
1. Use `ls()` to see available files
2. Read `/input/paper_info.md` for paper metadata
3. Read `/input/references.md` (or references_part_*.md) to get the reference list
4. Create a TODO list to track progress
5. For each reference, delegate searches to sub-agents for parallel processing (up to {self.max_concurrent_agents} at a time)
6. Validate each reference against search results
7. Save results to `/validation/` directory
8. Generate a final report at `/reports/validation_report.md`

Start by using `ls()` to see the pre-loaded files."""

        # Run the agent
        try:
            # Invoke the LangGraph agent with pre-loaded files
            initial_input = {
                "messages": [HumanMessage(content=task)],
                "files": state.get("files", {}),  # Pass pre-loaded PDF chunks
                "todos": state.get("todos", []),
            }
            result = self.agent.invoke(
                initial_input,
                {"recursion_limit": 300}  # 높은 recursion_limit - 많은 레퍼런스 처리용
            )

            # Extract results from the agent output
            messages = result.get("messages", [])
            output = ""
            if messages:
                # Get the last AI message
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        output = msg.content
                        break

            # Get final state including files created during processing
            final_files = result.get("files", {})
            final_todos = result.get("todos", [])

            # Build result dictionary
            validation_result = {
                "paper_path": paper_path,
                "output": output,
                "messages": messages,
                "files": final_files,  # Virtual filesystem contents
                "todos": final_todos,  # Final TODO state
                "state": state,  # Original state for reference
                "success": True,
            }

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Validation Complete")
                print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}\n")

            return validation_result

        except Exception as e:
            if self.verbose:
                print(f"\n⚠️ Validation Error: {str(e)}")

            return {
                "paper_path": paper_path,
                "output": f"Error: {str(e)}",
                "state": state,
                "success": False,
                "error": str(e),
            }

    def process_task(self, task: str, initial_files: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process a custom task with the Deep Agent.

        Args:
            task: Task description for the agent
            initial_files: Optional initial files for virtual filesystem

        Returns:
            Dictionary containing task results
        """
        try:
            # Prepare initial input with state
            initial_input = {
                "messages": [HumanMessage(content=task)],
                "files": initial_files or {},
                "todos": [],
            }
            result = self.agent.invoke(
                initial_input,
                {"recursion_limit": 300}
            )

            # Extract output from messages
            messages = result.get("messages", [])
            output = ""
            if messages:
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content:
                        output = msg.content
                        break

            return {
                "task": task,
                "output": output,
                "messages": messages,
                "files": result.get("files", {}),
                "todos": result.get("todos", []),
                "success": True,
            }

        except Exception as e:
            return {
                "task": task,
                "output": f"Error: {str(e)}",
                "success": False,
                "error": str(e),
            }


def create_deep_agent_supervisor(
    model: Optional[BaseChatModel] = None,
    tavily_api_key: Optional[str] = None,
    verbose: bool = True,
) -> DeepAgentSupervisor:
    """Factory function to create a Deep Agent Supervisor.

    Args:
        model: LLM to use (uses default from config if not provided)
        tavily_api_key: Tavily API key
        verbose: Whether to print detailed logs

    Returns:
        Configured DeepAgentSupervisor instance
    """
    if model is None:
        # Import config and create default model
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from config import settings
        model = settings.get_parser_llm()

    return DeepAgentSupervisor(
        model=model,
        tavily_api_key=tavily_api_key,
        verbose=verbose,
    )


if __name__ == "__main__":
    print("=== Deep Agent Supervisor 테스트 ===")

    # Test creating supervisor (without running)
    print("\n1. Supervisor 생성 테스트:")

    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import settings

    try:
        model = settings.get_parser_llm()
        supervisor = DeepAgentSupervisor(
            model=model,
            verbose=True,
        )
        print(f"   Tools 수: {len(supervisor.tools)}")
        print(f"   Tool 이름: {[t.name for t in supervisor.tools]}")
        print("   ✅ Supervisor 생성 성공")

    except Exception as e:
        print(f"   ⚠️ 생성 오류: {e}")
        print("   (이것은 LLM 연결이 없어서 정상입니다)")
