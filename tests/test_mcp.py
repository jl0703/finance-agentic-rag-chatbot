import pytest
from app.chatbot.chat.services.mcp_client import MCPClient

@pytest.mark.asyncio
async def test_get_tools():
    mcp_client = MCPClient()
    tools = await mcp_client.get_tools()
    assert isinstance(tools, list)