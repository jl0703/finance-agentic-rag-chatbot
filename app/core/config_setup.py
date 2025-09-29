from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
import redis.asyncio as redis

# Initialize Azure OpenAI chat model
CHAT_MODEL = AzureChatOpenAI(
    openai_api_key=settings.AZURE_OPENAI_API_KEY.get_secret_value(),
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    max_tokens=settings.MAX_TOKENS,
    temperature=settings.TEMPERATURE,
    seed=settings.SEED,
    streaming=settings.STREAMING,
)

# Initialize Azure OpenAI embedding model
EMBEDDING_MODEL = AzureOpenAIEmbeddings(
    openai_api_key=settings.AZURE_OPENAI_API_KEY.get_secret_value(),
    openai_api_version=settings.AZURE_OPENAI_EMBEDDING_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
)

# Initialize Qdrant Client
QDRANT_CLIENT = AsyncQdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY.get_secret_value(),
)

# Initialize Redis Semantic Cache
REDIS_CACHE = redis.from_url(settings.REDIS_URL)

# MCP Server Configurations
MCP_SERVERS = {
    "yfinance_mcp": {
        "command": "uv",
        "args": [
            "--directory",
            "C:\\Users\\User\\Documents\\mcp\\yfinance",  # Adjust path based on your local setup
            "run",
            "server.py",
        ],
        "transport": "stdio",
    },
    "tavily-remote": {
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.TAVILY_API_KEY.get_secret_value()}",
        "transport": "streamable_http",
    },
}
