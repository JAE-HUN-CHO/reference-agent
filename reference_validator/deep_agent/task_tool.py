"""
Task delegation tools for context isolation through sub-agents.

Based on teddynote-lab/deep-agents-from-scratch task_tool.py

This module provides the core infrastructure for creating and managing sub-agents
with isolated contexts. Sub-agents prevent context clash by operating with clean
context windows containing only their specific task description.
"""

from typing import Annotated, NotRequired, Optional, List, Callable
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command

from .prompts import RESEARCHER_INSTRUCTIONS, VALIDATOR_INSTRUCTIONS, get_current_time
from .state import DeepAgentState


class SubAgent(TypedDict):
    """Configuration for a specialized sub-agent."""
    name: str
    description: str
    prompt: str
    tools: NotRequired[List[str]]


def create_simple_sub_agent(
    model: BaseChatModel,
    tools: List[BaseTool],
    system_prompt: str,
) -> Callable:
    """Create a simple sub-agent that processes tasks.
    
    Args:
        model: LLM to use for the sub-agent
        tools: List of tools available to the sub-agent
        system_prompt: System prompt for the sub-agent
        
    Returns:
        Callable that processes tasks and returns results
    """
    # Create the agent using LangGraph's create_react_agent
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_prompt,
    )
    
    def run_sub_agent(state: dict) -> dict:
        """Execute the sub-agent with the given state."""
        messages = state.get("messages", [])
        files = state.get("files", {})
        
        # Get the last human message as the task
        task = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                task = msg.content
                break
        
        if not task:
            return {
                "messages": [HumanMessage(content="No task provided")],
                "files": files,
            }
        
        try:
            # Run the agent
            result = agent.invoke({
                "messages": messages,
            })
            
            # Get output from result messages
            result_messages = result.get("messages", [])
            output_content = ""
            
            for msg in reversed(result_messages):
                if hasattr(msg, 'content') and msg.content:
                    output_content = msg.content
                    break
            
            # Return the output
            return {
                "messages": messages + [HumanMessage(content=output_content or "Task completed")],
                "files": files,
            }
            
        except Exception as e:
            return {
                "messages": messages + [HumanMessage(content=f"Sub-agent error: {str(e)}")],
                "files": files,
            }
    
    return run_sub_agent


def create_task_tool(
    tools: List[BaseTool],
    subagents: List[SubAgent],
    model: BaseChatModel,
    state_schema: type = DeepAgentState,
):
    """Create a task delegation tool that enables context isolation through sub-agents.

    This function implements the core pattern for spawning specialized sub-agents with
    isolated contexts, preventing context pollution from the parent agent's conversation history.

    Args:
        tools: List of available tools that can be assigned to sub-agents
        subagents: List of specialized sub-agent configurations
        model: The language model to use for all agents
        state_schema: The state schema (typically DeepAgentState)

    Returns:
        A 'task' tool that can delegate work to specialized sub-agents
    """
    # Build sub-agent registry
    agents = {}
    
    # Map tools by name
    tools_by_name = {}
    for tool_ in tools:
        if isinstance(tool_, BaseTool):
            tools_by_name[tool_.name] = tool_
        else:
            # If it's a function, convert to tool
            tools_by_name[tool_.__name__] = tool_
    
    # Create sub-agents
    for _agent in subagents:
        # Get tools for this sub-agent
        if "tools" in _agent and _agent["tools"]:
            _tools = [tools_by_name[t] for t in _agent["tools"] if t in tools_by_name]
        else:
            _tools = list(tools_by_name.values())
        
        # Create simple callable sub-agent
        agents[_agent["name"]] = create_simple_sub_agent(
            model,
            _tools,
            _agent["prompt"],
        )
    
    # Build description of available sub-agents
    other_agents_string = "\n".join([
        f"- {_agent['name']}: {_agent['description']}" for _agent in subagents
    ])
    
    task_description = f"""Delegate a task to a specialized sub-agent with isolated context.

Available agents for delegation:
{other_agents_string}

Use this tool when you need to:
- Search for academic papers (use research-agent)
- Validate references against search results (use validator-agent)
- Perform any task that benefits from focused, isolated processing

Each sub-agent operates with a clean context window containing only the task description,
preventing context pollution from the parent agent's conversation history.
"""
    
    @tool(description=task_description)
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Delegate a task to a specialized sub-agent with isolated context.

        This creates a fresh context for the sub-agent containing only the task description,
        preventing context pollution from the parent agent's conversation history.
        
        Args:
            description: Clear, specific task or research question for the sub-agent
            subagent_type: Type of agent to use (e.g., "research-agent", "validator-agent")
            state: Current agent state (injected)
            tool_call_id: Tool call identifier (injected)
            
        Returns:
            Command with sub-agent results and file updates
        """
        # Validate sub-agent type
        if subagent_type not in agents:
            error_msg = f"Error: Unknown agent type '{subagent_type}'. Available types: {list(agents.keys())}"
            return Command(
                update={
                    "messages": [ToolMessage(error_msg, tool_call_id=tool_call_id)],
                }
            )
        
        # Get the sub-agent
        sub_agent = agents[subagent_type]
        
        # Create isolated context with only the task description
        # Sub-agent doesn't see parent's conversation history
        new_state = {
            "messages": [HumanMessage(content=description)],
            "files": state.get("files", {}),
            "todos": state.get("todos", []),
        }
        
        try:
            # Execute sub-agent
            result = sub_agent(new_state)
            
            # Get the sub-agent's response
            response_messages = result.get("messages", [])
            sub_agent_response = ""
            
            # Get the last message as the response
            if response_messages:
                last_msg = response_messages[-1]
                if hasattr(last_msg, 'content'):
                    sub_agent_response = last_msg.content
                else:
                    sub_agent_response = str(last_msg)
            
            # Format response
            formatted_response = f"""📋 Sub-agent ({subagent_type}) completed task:

**Task:** {description[:200]}...

**Result:**
{sub_agent_response[:2000]}

📁 Files in system: {list(result.get('files', {}).keys())[:5]}"""
            
            return Command(
                update={
                    "files": result.get("files", {}),
                    "messages": [ToolMessage(formatted_response, tool_call_id=tool_call_id)],
                }
            )
            
        except Exception as e:
            error_response = f"⚠️ Sub-agent ({subagent_type}) error: {str(e)}"
            return Command(
                update={
                    "messages": [ToolMessage(error_response, tool_call_id=tool_call_id)],
                }
            )
    
    return task


# Default sub-agent configurations for reference validation
DEFAULT_RESEARCH_SUBAGENT = SubAgent(
    name="research-agent",
    description="Delegate reference search tasks. Give this agent a specific reference to search for (title, authors, year).",
    prompt=RESEARCHER_INSTRUCTIONS.format(date=get_current_time()),
    tools=["tavily_search", "think_tool"],
)

DEFAULT_VALIDATOR_SUBAGENT = SubAgent(
    name="validator-agent",
    description="Delegate reference validation tasks. Give this agent search results and reference details to validate.",
    prompt=VALIDATOR_INSTRUCTIONS,
    tools=["think_tool", "read_file", "write_file"],
)


def get_default_subagents() -> List[SubAgent]:
    """Get default sub-agent configurations for reference validation.
    
    Returns:
        List of default sub-agent configurations
    """
    return [
        DEFAULT_RESEARCH_SUBAGENT,
        DEFAULT_VALIDATOR_SUBAGENT,
    ]


if __name__ == "__main__":
    print("=== Task Tool 테스트 ===")
    
    # Test SubAgent definition
    print("\n1. SubAgent 정의 테스트:")
    research_agent: SubAgent = {
        "name": "research-agent",
        "description": "Searches for academic papers",
        "prompt": "You are a research assistant...",
        "tools": ["tavily_search", "think_tool"],
    }
    print(f"   Name: {research_agent['name']}")
    print(f"   Tools: {research_agent.get('tools', [])}")
    
    # Test default subagents
    print("\n2. Default Subagents:")
    for agent in get_default_subagents():
        print(f"   - {agent['name']}: {agent['description'][:50]}...")
