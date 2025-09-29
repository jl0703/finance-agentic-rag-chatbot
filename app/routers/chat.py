import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.chatbot.chat.workflow.graph import ChatOrchestrator
from app.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post("/", status_code=status.HTTP_200_OK)
async def chat_response(
    request: ChatRequest,
    orchestrator: ChatOrchestrator = Depends(ChatOrchestrator),
) -> ChatResponse:
    """
    Generate response for the given chat message.

    Args:
        request (ChatRequest): Chat request containing user message.

    Returns:
        ChatResponse: Generated response from the chatbot.

    Raises:
        HTTPException: If there is an error generating the response, returns a 500 error.
    """
    try:
        logger.info(f"Processing chat request")
        graph = orchestrator.build_graph()
        output = await graph.ainvoke(
            {"user_id": request.user_id, "message": request.message},
            config={"recursion_limit": 10},
        )

        logger.info("Response generated successfully")

        return ChatResponse(response=output["response"])
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


@router.post("/stream", status_code=status.HTTP_200_OK)
async def stream_chat_response(
    request: ChatRequest,
    orchestrator: ChatOrchestrator = Depends(ChatOrchestrator),
) -> StreamingResponse:
    """
    Stream response for the given chat message in real-time.

    Args:
        request (ChatRequest): Chat request containing user message.

    Returns:
        StreamingResponse: Real-time streaming response from the chatbot.

    Raises:
        HTTPException: If there's an error initiating or streaming the response, returns a 500 error.
    """
    try:
        logger.info("Processing streaming chat request")

        async def generate_stream() -> AsyncGenerator[str, None]:
            """
            Async generator yielding response chunks in Server-Sent Events format.
            """
            try:
                async for chunk in orchestrator.build_graph().astream(
                    {"user_id": request.user_id, "message": request.message}
                ):
                    for node, update in chunk.items():
                        if isinstance(update, dict) and "response" in update and update["response"]:
                            yield f"{update['response']}\n"

            except Exception as e:
                logger.error(f"Error in streaming: {str(e)}", exc_info=True)
                yield f"Error: {str(e)}\n"

        logger.info("Streaming response initiated")

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    except Exception as e:
        logger.error(f"Error initiating streaming: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initiate streaming response",
        )
