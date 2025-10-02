import redis.asyncio as redis
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from qdrant_client import AsyncQdrantClient
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import AzureOpenAITextVectorizer

from app.core.config import settings

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

# Initialize Redis
REDIS = redis.from_url(settings.REDIS_URL)

# Initialize Azure OpenAI Text Vectorizer
API_CONFIG = {
    "azure_endpoint": settings.AZURE_OPENAI_ENDPOINT,
    "api_key": settings.AZURE_OPENAI_API_KEY.get_secret_value(),
    "api_version": settings.AZURE_OPENAI_EMBEDDING_API_VERSION
}

AZURE_VECTORIZER = AzureOpenAITextVectorizer(
    model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    api_config=API_CONFIG,
    dtype=settings.VECTORIZER_DTYPE,
)

# Initialize Redis Semantic Cache
SEMANTIC_CACHE = SemanticCache(
    name=settings.CACHE_NAME,
    redis_url=settings.REDIS_URL,
    ttl=settings.CACHE_TTL,                       
    distance_threshold=settings.CACHE_DISTANCE_THRESHOLD,
    vectorizer=AZURE_VECTORIZER,
)

# MCP Server Configurations
MCP_SERVERS = {
    "yfinance_mcp": {
        "command": "docker",
        "args": ["run", "-i", "--rm", "yfinance-mcp"],
        "transport": "stdio",
    },
    "tavily-remote": {
        "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={settings.TAVILY_API_KEY.get_secret_value()}",
        "transport": "streamable_http",
    },
}
