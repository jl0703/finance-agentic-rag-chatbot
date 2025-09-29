import logging
import uuid
from typing import List, Optional

from langchain.schema import Document

from app.core.config import settings
from app.core.config_setup import EMBEDDING_MODEL, QDRANT_CLIENT

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant Vector Store for managing document embeddings."""

    def __init__(self):
        """Initialize the Qdrant vector store with configuration settings."""
        self.collection_name = settings.COLLECTION_NAME
        self.qdrant_client = QDRANT_CLIENT
        self.embedding_model = EMBEDDING_MODEL

    async def create_collection(self) -> bool:
        """
        Creates the collection if it doesn't exist, or verifies it exists
        with the correct vector configuration.

        Returns:
            bool: True if collection exists or was created successfully,
                  False otherwise.

        Raises:
            Exception: If there's an error creating or verifying the collection.
        """
        try:
            collections = await self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                await self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={"size": 1536, "distance": "Cosine"},
                )
                logger.info(f"Created collection: {self.collection_name}")

            else:
                logger.info(f"Collection {self.collection_name} already exists")

            return True

        except Exception as e:
            logger.error(f"Error creating collection: {e}", exc_info=True)
            return False

    async def add_documents(self, documents: List[Document]) -> bool:
        """
        Processes a list of Document and stores them in the Qdrant collection
        with their embeddings.

        Args:
            documents (List[Document]): List of LangChain Document objects
                                       to add to the vector store.

        Returns:
            bool: True if documents were added successfully, False otherwise.

        Raises:
            ValueError: If documents list is empty or contains invalid documents.
            Exception: If there's an error during document addition.
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        try:
            logger.info(
                f"Adding {len(documents)} documents to collection {self.collection_name}"
            )

            for doc in documents:
                embedding = await self.embedding_model.aembed_query(doc.page_content)

                await self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        {
                            "id": str(uuid.uuid4()),
                            "vector": embedding,
                            "payload": {
                                "content": doc.page_content,
                                "metadata": doc.metadata,
                            },
                        }
                    ],
                )

            logger.info(f"Successfully added {len(documents)} documents")

            return True

        except Exception as e:
            logger.error(f"Error adding documents: {e}", exc_info=True)
            return False

    async def similarity_search(
        self, query: str, k: Optional[int] = 5
    ) -> List[Document]:
        """
        Performs a similarity search using the provided query string
        and returns the top-k most similar documents.

        Args:
            query (str): The search query string.
            k (int): Number of similar documents to return. Defaults to 5.

        Returns:
            List[Document]: List of similar documents, sorted by relevance.

        Raises:
            ValueError: If query is empty or k is less than 1.
            Exception: If there's an error during the search operation.
        """
        if not query:
            raise ValueError("Query cannot be empty")

        try:
            query_embedding = await self.embedding_model.aembed_query(query.strip())

            search_result = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=0.8,
            )

            documents = []

            for result in search_result:
                doc = Document(
                    page_content=result.payload.get("content", ""),
                    metadata=result.payload.get("metadata", {}),
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error in similarity search: {e}", exc_info=True)
            return []

    async def delete_collection(self) -> bool:
        """
        Delete the Qdrant collection.

        Permanently removes the collection and all its data.
        Use with caution as this action cannot be undone.

        Returns:
            bool: True if collection was deleted successfully, False otherwise.

        Raises:
            Exception: If there's an error deleting the collection.
        """
        try:
            await self.qdrant_client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted successfully")

            return True

        except Exception as e:
            logger.error(f"Error deleting collection: {e}", exc_info=True)
            return False
