import logging
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.chatbot.ingestion.workflow.graph import DocumentIngestionOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
)


@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload_document(file: UploadFile):
    """
    Upload a document to the vector store.

    Args:
        file (UploadFile): The file to upload to the vector store.
    """
    temp_file_path = None
    orchestrator = DocumentIngestionOrchestrator()

    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        logger.info(f"Uploaded file saved to temporary path: {temp_file_path}")

        graph = orchestrator.build_graph()
        logger.info("State graph built successfully.")

        response = await graph.ainvoke({"file_path": temp_file_path})
        logger.info(f"Ingestion completed successfully: {response}")

        return response

    except Exception as e:
        logger.error(f"Error during document upload & ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during ingestion."
        )
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.debug(f"Temporary file cleaned up: {temp_file_path}")
