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
from .file_tools import ls, read_file, write_file, save_to_file
from .todo_tools import write_todos, read_todos, create_todos, format_todos_for_display
from .research_tools import think_tool, create_tavily_search_tool
from .task_tool import create_task_tool, SubAgent, get_default_subagents
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
            # File system tools
            ls,
            read_file,
            write_file,
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
        agent = create_react_agent(
            model=self.model,
            tools=self.tools,
            prompt=system_prompt,
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
        
        # Create initial state
        state = create_initial_deep_state(paper_path)
        
        # Build the task description
        if max_references:
            task = f"""Please validate the references in the paper at: {paper_path}

Validate up to {max_references} references.

Your workflow should be:
1. Create a TODO list to track progress
2. Parse the PDF to extract references
3. For each reference, search to verify it exists
4. Validate each reference against search results
5. Save results to the virtual file system
6. Generate a final validation report

Use the virtual file system to store:
- Parsed references (input/references.md)
- Search results (search/ref_[id].md)
- Validation results (validation/ref_[id].md)
- Final report (reports/validation_report.md)

Start by creating your TODO list and then proceed with the validation."""
        else:
            task = f"""Please validate ALL references in the paper at: {paper_path}

Your workflow should be:
1. Create a TODO list to track progress
2. Parse the PDF to extract references
3. For each reference, delegate searches to sub-agents for parallel processing
4. Validate each reference against search results
5. Save results to the virtual file system
6. Generate a comprehensive final validation report

Use sub-agent delegation to process multiple references in parallel (up to {self.max_concurrent_agents} at a time).

Use the virtual file system to store:
- Parsed references (input/references.md)
- Search results (search/ref_[id].md)  
- Validation results (validation/ref_[id].md)
- Final report (reports/validation_report.md)

Start by creating your TODO list and then proceed with the validation."""
        
        # Run the agent
        try:
            # Invoke the LangGraph agent (높은 recursion_limit 설정 - 많은 레퍼런스 처리용)
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=task)]},
                {"recursion_limit": 300}
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
            
            # Build result dictionary
            validation_result = {
                "paper_path": paper_path,
                "output": output,
                "messages": messages,
                "state": state,
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
    
    def process_task(self, task: str) -> Dict[str, Any]:
        """Process a custom task with the Deep Agent.
        
        Args:
            task: Task description for the agent
            
        Returns:
            Dictionary containing task results
        """
        try:
            result = self.agent.invoke(
                {"messages": [HumanMessage(content=task)]},
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
