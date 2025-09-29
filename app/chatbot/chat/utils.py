from typing import List, Optional, Type

from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from app.core.config_setup import CHAT_MODEL


def build_chain(system_template: str, schema: Type[BaseModel]):
    """
    Builds a simple LLM chain with system template.

    Args:
        system_template (str): String containing the system prompt template.
        schema (Type[BaseModel]): The Pydantic model schema for structured output.

    Returns:
        A LangChain sequential chain that processes the input through the model.
    """
    prompt = ChatPromptTemplate.from_messages([("system", system_template)])

    return prompt | CHAT_MODEL.with_structured_output(schema)


def build_agent(
    system_template: str,
    tools: Optional[List[BaseTool]] = None,
):
    """
    Builds a ReAct agent that can use tools.

    The agent follows the ReAct (Reasoning + Acting) pattern, allowing it to:
    1. Think about what tools to use
    2. Call tools to gather information
    3. Reason about the results to answer the query

    Args:
        system_template (str): String containing the system prompt template.
        tools (Optional[List[BaseTool]]): Optional list of BaseTool objects that the agent can use.

    Returns:
        A LangGraph ReAct agent that can be invoked.
    """

    return create_react_agent(CHAT_MODEL, prompt=system_template, tools=tools or [])


def extract_tool_calls(messages: dict) -> tuple[list[str], str]:
    """
    Extracts tool call names and the final assistant response from agent output.

    Args:
        messages (dict): The agent output containing a 'messages' list.

    Returns:
        tuple[list[str], str]: (list of tool names used, final assistant message content)
    """
    tool_calls = []
    for m in messages:
        if hasattr(m, "additional_kwargs") and "tool_calls" in m.additional_kwargs:
            tool_calls.extend(m.additional_kwargs["tool_calls"])

    tool_names = [tool["function"]["name"] for tool in tool_calls]

    assistant_msgs = [
        m
        for m in messages
        if m.__class__.__name__ == "AIMessage"
        and m.content
        and not m.content.strip() == ""
    ]
    final_content = assistant_msgs[-1].content if assistant_msgs else None

    return tool_names, final_content
