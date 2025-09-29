from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.logging_config import setup_logging
from app.routers import chat, health, ingestion

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application starting up")
    logger.info("CORS middleware enabled")

    yield

    # Shutdown
    logger.info("Application shutting down")


app = FastAPI(
    title="Finance Agentic RAG Chatbot",
    description="An agentic RAG chatbot for financial and investment queries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(ingestion.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)