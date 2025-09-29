import json
import logging
from typing import List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from app.core.config_setup import MCP_SERVERS

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages connections to multiple MCP servers and provides tools to the agent."""

    def __init__(self):
        """Initialize the MCP client with server configurations."""
        self.client = MultiServerMCPClient(MCP_SERVERS)

    async def get_tools(self) -> List[BaseTool]:
        """
        Retrieves tools from the MCP servers.

        Returns:
            List[BaseTool]: A list of tools available on the MCP servers.
        """
        try:
            tools = await self.client.get_tools()
            logger.info("[MCPManager] Loaded %d tools from MCP servers", len(tools))

            return tools
        except Exception as e:
            logger.exception(
                "[MCPManager] Failed to load tools from MCP servers: %s", str(e)
            )
            return []

    async def get_tools_from(self, server_name: str) -> List[BaseTool]:
        """
        Retrieves tools from a specific MCP server.

        Args:
            server_name (str): The name of the MCP server.

        Returns:
            List[BaseTool]: A list of tools available on the specified MCP server.
        """
        try:
            async with self.client.session(server_name) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                logger.info(
                    "[MCPManager] Loaded %d tools from server %s",
                    len(tools),
                    server_name,
                )

                return tools
        except Exception as e:
            logger.exception(
                "[MCPManager] Failed to load tools from MCP server '%s': %s",
                server_name,
                str(e),
            )
            return []

    async def get_tools_json(self) -> str:
        """
        Get a JSON summary of the tools fetched from MCP servers.

        Returns:
            str: A JSON string representation of the tools.
        """
        try:
            tools = await self.client.get_tools()

            tools_dict = [{"name": t.name, "description": t.description} for t in tools]

            return json.dumps(tools_dict, indent=2)
        except Exception as e:
            logger.exception(
                "[MCPManager] Failed to get tools JSON from MCP servers: %s", str(e)
            )
            return "[]"
