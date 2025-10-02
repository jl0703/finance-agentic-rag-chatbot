import logging

from fastapi import APIRouter, Depends, HTTPException, status

from typing import Any
from app.chatbot.chat.services.mcp_client import MCPClient
from app.chatbot.chat.services.openai_client import OpenAIClient
from app.chatbot.ingestion.services.vector_store import QdrantVectorStore
from app.core.config_setup import REDIS
from app.chatbot.chat.services.openai_client import OpenAIClient
from app.core.config_setup import REDIS

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get("/openai", status_code=status.HTTP_200_OK)
async def openai_health(openai_client: OpenAIClient = Depends(OpenAIClient)) -> dict[str, str]:
    """
    Check Azure OpenAI API connectivity by sending a minimal test prompt.

    Returns:
        dict[str, str]: Health status information.

    Raises:
        HTTPException: 503 Service Unavailable if the OpenAI API cannot be reached.
    """
    try:
        response = await openai_client.generate_response("Hi")

        if response:
            return {"status": "Healty"}
        else:
            raise Exception("Empty response from Azure OpenAI API")
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"OpenAI API unhealthy: {str(e)}",
        )


@router.get("/redis", status_code=status.HTTP_200_OK)
async def redis_health() -> dict[str, str]:
    """
    Check redis connectivity. Pings the redis with a simple ping.

    Returns:
        dict[str, str]: Health status information.

    Raises:
        HTTPException: 503 Service Unavailable if the database cannot be reached.
    """
    try:
        cache = REDIS
        await cache.ping()

        return {"status": "Healthy"}
    except Exception as e:
        logger.error(f"Redis health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis unreachable: {str(e)}",
        )


@router.get("/qdrant", status_code=status.HTTP_200_OK)
async def db_vector_store(vdb: QdrantVectorStore = Depends(QdrantVectorStore)) -> dict[str, str]:
    """
    Check the health status of the Qdrant vector store.

    Returns:
        dict[str, str]: Health status information

    Raises:
        HTTPException: 503 Service Unavailable if the Qdrant vector store cannot be reached.
    """
    try:
        await vdb.create_collection()

        return {"status": "Healthy"}
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant vector store unhealthy: {str(e)}",
        )

@router.get("/mcp", status_code=status.HTTP_200_OK)
async def mcp_health(mcp: MCPClient = Depends(MCPClient)) -> dict[str, Any]:
    """
    Check health status of MCP servers.

    Returns:
        dict[str, Any]: Health status information
        
    Raises:
        HTTPException: 503 Service Unavailable if any MCP server is unreachable.
    """
    try:
        tools = await mcp.get_tools()

        if not tools:
            return {"status": "No tools available.", "tool_count": 0}

        return {"status": "Healthy", "tool_count": len(tools)}
    except Exception as e:
        logger.error(f"MCP health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MCP servers unhealthy: {str(e)}",
        )
