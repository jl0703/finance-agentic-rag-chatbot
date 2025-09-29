from pydantic import Field, SecretStr, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Azure Chat OpenAI Configuration
    AZURE_OPENAI_API_KEY: SecretStr = Field(..., repr=False)
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_DEPLOYMENT: str
    AZURE_OPENAI_API_VERSION: str
    MAX_TOKENS: int = 16000
    TEMPERATURE: int = 0
    SEED: int = 42
    STREAMING: bool = True

    # Azure Embedding Configuration
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str
    AZURE_OPENAI_EMBEDDING_API_VERSION: str

    # Tavily Search Configuration
    TAVILY_API_KEY: SecretStr = Field(..., repr=False)

    # Cache
    REDIS_URL: str

    # Vector Database
    QDRANT_URL: str
    QDRANT_API_KEY: SecretStr = Field(..., repr=False)
    COLLECTION_NAME: str = "documents"

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = ConfigDict(env_file=".env")


settings = Settings()
