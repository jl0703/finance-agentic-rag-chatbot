import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.chatbot.chat.schemas.model import Plan, Supervisor
from app.chatbot.chat.schemas.state import InputState, OutputState, OverallState
from app.chatbot.chat.services.mcp_client import MCPClient
from app.chatbot.chat.services.openai_client import OpenAIClient
from app.chatbot.chat.services.redis_cache import RedisCache
from app.chatbot.chat.templates import (
    generate_factual_response,
    planning_prompt,
    supervision_prompt,
)
from app.chatbot.chat.utils import build_agent, build_chain, extract_tool_calls
from app.chatbot.ingestion.services.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Orchestrator for the chat workflow."""

    def __init__(self):
        """Initialize the orchestrator and its service dependencies."""
        self.openai_client = OpenAIClient()
        self.mcp_manager = MCPClient()
        self.vector_store = QdrantVectorStore()
        self.cache = RedisCache()

    async def planner(self, state: InputState) -> OverallState:
        """
        Analyze the user message and generate a step-by-step plan for the workflow.

        Args:
            state (InputState): The input state containing the user's message.

        Returns:
            OverallState: The updated state including the original message, generated plan, and available tools.
        """
        try:
            logger.info("[Planner] Planning next steps.")
            
            cached = self.cache.get_cached(state["message"])
            
            if cached:
                logger.info("[Planner] Cache hit. Returning cached response.")
                
                return {
                    "response": cached["response"],
                    "tools_used": cached["metadata"].get("tools_called", []),
                    "is_cached": True,
                }

            tools_json = await self.mcp_manager.get_tools_json()
            chain = build_chain(planning_prompt(), Plan)
            response = await chain.ainvoke(
                {"query": state["message"], "tools": tools_json}
            )

            logger.info("[Planner] Generated plan: %s", response)

            return {
                "message": state["message"],
                "plan": response.steps,
                "tools": tools_json,
            }
        except Exception as e:
            logger.exception("[Planner] Error during planning: %s", str(e))
            raise

    async def supervisor(self, state: OverallState) -> Command[Literal["retrieval", "generator", "__end__"]]:
        """
        Supervise and decide the next workflow node to execute based on the current state.

        Args:
            state (OverallState): The current workflow state, including plan.

        Returns:
            Command[Literal["retrieval", "generator", "__end__"]] : The command to transition to the next node.
        """
        try:
            logger.info("[Supervisor] Supervising workflow.")
            chain = build_chain(supervision_prompt(), schema=Supervisor)
            response = await chain.ainvoke(
                {
                    "query": state["message"],
                    "plan": state.get("plan", []),
                    "docs": state.get("retrieved_docs", "No documents."),
                    "response": state.get("response", "No response so far."),
                }
            )

            logger.info("[Supervisor] Next node: %s", response)

            return Command(goto=response.next_node)
        except Exception as e:
            logger.exception("[Supervisor] Error during supervision: %s", str(e))
            return Command(
                goto="__end__",
                update={
                    "response": "Sorry, I encountered an error while supervising the workflow."
                },
            )

    async def retrieval(self, state: OverallState) -> Command[Literal["supervisor"]]:
        """
        Retrieve relevant documents from the vector store based on the user's message.

        Args:
            state (OverallState): The current workflow state, including the user message.

        Returns:
            Command[Literal["supervisor"]]: The command to transition to the supervisor node with retrieved documents.
        """
        try:
            logger.info("[Retrieval] Retrieving docs.")
            retrieved_docs = await self.vector_store.similarity_search(state["message"])
            logger.info("[Retrieval] Retrieved %d documents", len(retrieved_docs))

            if len(retrieved_docs) == 0:
                return Command(
                    goto="supervisor",
                    update={"retrieved_docs": "No documents found for the query."},
                )

            formatted_docs = "\n\n".join(
                [
                    f"Document {i + 1}: \n{doc.page_content}"
                    for i, doc in enumerate(retrieved_docs)
                ]
            )

            return Command(goto="supervisor", update={"retrieved_docs": formatted_docs})
        except Exception as e:
            logger.exception("[Retrieval] Error retrieving docs: %s", e)
            return Command(goto="supervisor", update={"retrieved_docs": ""})

    async def generator(self, state: OverallState) -> Command[Literal["supervisor"]]:
        """
        Generate a response to the user's query using the retrieved documents and available tools.

        Args:
            state (OverallState): The current workflow state, including the user message and retrieved docs.

        Returns:
            Command[Literal["supervisor"]]: The command to transition to the supervisor node with the generated response.
        """
        try:
            logger.info("[Generator] Generating response.")

            tools = await self.mcp_manager.get_tools()
            system_template = generate_factual_response().format(
                query=state["message"],
                tools=state.get("tools", []),
                docs=state.get("retrieved_docs", "No documents."),
            )
            output = await build_agent(system_template, tools).ainvoke(
                {"messages": state["message"]}
            )

            tool_names, final_output = extract_tool_calls(output["messages"])
            
            self.cache.store(state["message"], final_output, {"tools_called": tool_names})

            logger.info("[Generator] Response generated.")

            return Command(
                goto="supervisor",
                update={"response": final_output, "tools_used": tool_names},
            )
        except Exception as e:
            logger.exception("[Generator] Error generating output for state: %s", e)
            return Command(
                goto="supervisor",
                update={
                    "response": f"Sorry, I encountered an error while generating the response: {str(e)}",
                    "tools_used": [],
                },
            )
        
    def planner_route(self, state: OverallState) -> Literal["supervisor", "__end__"]:
        """
        Determine the next step after planning.
        
        Returns:
            Literal["supervisor", "__end__"]: The next step after planning.
        """
        if state.get("is_cached") == True:
            return "__end__"

        return "supervisor"
        

    def build_graph(self) -> StateGraph:
        """
        Build the graph for the chat orchestrator.

        Returns:
            StateGraph: The graph for the chat orchestrator.
        """
        graph = StateGraph(
            OverallState, input_schema=InputState, output_schema=OutputState
        )

        graph.add_node("planner", self.planner)
        graph.add_node("supervisor", self.supervisor)
        graph.add_node("retrieval", self.retrieval)
        graph.add_node("generator", self.generator)

        graph.add_edge(START, "planner")
        graph.add_conditional_edges("planner", self.planner_route)
        graph.add_edge("supervisor", END)

        return graph.compile()
