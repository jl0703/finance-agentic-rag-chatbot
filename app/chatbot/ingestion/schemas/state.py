from typing import List, Optional, TypedDict

from langchain.schema import Document


class InputState(TypedDict):
    """Input state for document ingestion"""

    file_path: str


class DocumentState(TypedDict):
    """Document state for document ingestion"""

    documents: List[Document]


class ChunkState(TypedDict):
    """Chunk state for document ingestion"""

    chunks: List[Document]


class StoreState(TypedDict):
    """Store state for document ingestion"""

    stored_count: int
    error: Optional[str] = None


class OverallState(InputState, DocumentState, ChunkState, StoreState):
    """Overall state for document ingestion"""

    pass
