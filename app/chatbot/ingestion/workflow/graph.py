import logging
import os

from langgraph.graph import END, START, StateGraph

from app.chatbot.ingestion.schemas.state import (
    ChunkState,
    DocumentState,
    InputState,
    OverallState,
    StoreState,
)
from app.chatbot.ingestion.services.vector_store import QdrantVectorStore
from app.chatbot.ingestion.utils import (
    chunk_documents,
    docx_loader,
    html_loader,
    pdf_loader,
    txt_loader,
)

logger = logging.getLogger(__name__)


class DocumentIngestionOrchestrator:
    """Orchestrator for document ingestion workflow."""

    def __init__(self):
        """Initialize the orchestrator and its service dependencies."""
        self.qdrant_vector_store = QdrantVectorStore()

    async def doc_loader(self, state: InputState) -> DocumentState:
        """
        Load a document based on its file extension.

        Args:
            state (InputState): The input state containing the file path.

        Returns:
            DocumentState: A dictionary containing a list of Document objects.

        Raises:
            ValueError: If the file extension is unsupported.
        """
        path = state["file_path"]
        ext = os.path.splitext(path)[1]

        logger.info(f"Loading document from {path} with extension {ext}")

        try:
            if ext == ".pdf":
                docs = pdf_loader(path)
            elif ext == ".docx":
                docs = docx_loader(path)
            elif ext in (".html", ".htm"):
                docs = html_loader(path)
            elif ext == ".txt":
                docs = txt_loader(path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"Failed to load document: {e}")
            raise

        logger.info(f"Loaded {len(docs)} documents")
        return {"documents": docs}

    async def doc_chunker(self, state: DocumentState) -> ChunkState:
        """
        Chunk the loaded documents into smaller, manageable pieces.

        Args:
            state (DocumentState): The state containing the loaded documents.

        Returns:
            ChunkState: A dictionary containing a list of chunked Document objects.
        """
        documents = state["documents"]

        if not documents:
            logger.warning("No documents to chunk")
            return {"chunks": []}

        logger.info(f"Chunking {len(documents)} documents")
        chunks = chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        return {"chunks": chunks}

    async def store_chunks(self, state: ChunkState) -> StoreState:
        """
        Store the embedded chunked documents into a vector store.

        Args:
            state (ChunkState): The state containing the chunked documents.

        Returns:
            StoreState: A dictionary containing the count of stored chunks and any error message.
        """
        chunks = state["chunks"]

        if not chunks:
            logger.warning("No chunks to store")
            return {"stored_count": 0, "error": "No chunks to store"}

        try:
            await self.qdrant_vector_store.create_collection()

            is_success = await self.qdrant_vector_store.add_documents(chunks)

            if is_success:
                logger.info(f"Successfully stored {len(chunks)} chunks")
                return {"stored_count": len(chunks), "error": None}

            else:
                logger.error("Failed to store chunks")
                return {"stored_count": 0, "error": "Failed to store chunks"}

        except Exception as e:
            logger.error(f"Error storing chunks: {e}")
            return {"stored_count": 0, "error": str(e)}

    def build_graph(self) -> StateGraph:
        """
        Build the state graph for the document ingestion workflow.

        Returns:
            StateGraph: The compiled state graph representing the ingestion workflow.
        """
        graph = StateGraph(OverallState, input=InputState, output=StoreState)

        graph.add_node("doc_loader", self.doc_loader)
        graph.add_node("doc_chunker", self.doc_chunker)
        graph.add_node("store_chunks", self.store_chunks)

        graph.add_edge(START, "doc_loader")
        graph.add_edge("doc_loader", "doc_chunker")
        graph.add_edge("doc_chunker", "store_chunks")
        graph.add_edge("store_chunks", END)

        logger.info("State graph built successfully")

        return graph.compile()
