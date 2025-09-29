from typing import AsyncGenerator

from app.core.config_setup import CHAT_MODEL, EMBEDDING_MODEL


class OpenAIClient:
    """OpenAI client for chat and embedding services."""

    def __init__(self):
        """Initialize OpenAI clients for chat and embedding."""
        self.llm = CHAT_MODEL
        self.emb = EMBEDDING_MODEL

    async def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for the provided text from OpenAI embedding model.

        Args:
            text (str): The text to embed.

        Returns:
            list[float]: The embedding vector for the text.
        """
        try:
            return await self.emb.aembed_query(text)
        except Exception as e:
            raise Exception(f"The OpenAI model ran into an issue: {e}")

    async def generate_response(self, messages: list) -> str:
        """
        Get response from a Azure OpenAi chat model in non-streaming mode.

        Args:
            messages (list): List of messages to send to the chat model.
            schema (Type[BaseModel]): The Pydantic model schema for structured output.

        Returns:
            str: The content of the response message.
        """
        try:
            response = await self.llm.ainvoke(messages)
            return response
        except Exception as e:
            raise Exception(f"The OpenAI model ran into an issue: {e}")

    async def generate_streaming_response(
        self, messages: list
    ) -> AsyncGenerator[str, None]:
        """
        Get response from a OpenAI model in streaming mode.

        Args:
            messages (list): List of messages to send to the chat model.

        Returns:
            AsyncGenerator[str, None]: An asynchronous generator yielding chunks of the response content.
        """
        try:
            async for chunk in self.llm.astream(messages):
                yield chunk.content
        except Exception as e:
            raise Exception(f"The OpenAI model ran into an issue: {e}")
