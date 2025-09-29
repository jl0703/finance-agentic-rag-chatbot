import re
from pathlib import Path
from typing import List, Union

from langchain.schema import Document
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_docling import DoclingLoader
from markdownify import markdownify as md


def pdf_loader(path: Union[str, Path]) -> List[Document]:
    """
    Load a PDF earnings report, preserving page structure.

    Args:
        path (Union[str, Path]): File path to the PDF document.

    Returns:
        List[Document]: One document per page with markdown content and page metadata.
    """
    loader = PyMuPDFLoader(file_path=path, mode="page")
    docs = loader.load()

    markdown_docs = []

    for i, doc in enumerate(docs):
        markdown_content = _convert_to_markdown(doc.page_content)

        markdown_doc = Document(
            page_content=markdown_content,
            metadata={
                "source": str(path),
                "page_number": i + 1,
                "total_pages": len(docs),
                "file_type": "pdf",
                "content_format": "markdown",
            },
        )

        markdown_docs.append(markdown_doc)

    return markdown_docs


def docx_loader(path: Union[str, Path]) -> List[Document]:
    """
    Load a DOCX earnings report as markdown.

    Args:
        path (Union[str, Path]): File path of the DOCX document.

    Returns:
        List[Document]: One document with markdown content and metadata.
    """
    loader = DoclingLoader(file_path=path, export_type="MARKDOWN")
    docs = loader.load()

    markdown_docs = []

    for i, doc in enumerate(docs):
        markdown_doc = Document(
            page_content=doc.page_content,
            metadata={
                "source": str(path),
                "page_number": i + 1,
                "total_pages": len(docs),
                "file_type": "docx",
                "content_format": "markdown",
            },
        )

        markdown_docs.append(markdown_doc)

    return markdown_docs


def html_loader(path: Union[str, Path]) -> List[Document]:
    """
    Load a single HTML earnings report and convert to markdown.

    Args:
        path (Union[str, Path]): File path or URL of the HTML report.

    Returns:
        List[Document]: One document with markdown content and metadata.
    """
    loader = BSHTMLLoader(path)
    docs = loader.load()

    markdown_docs = []

    for i, doc in enumerate(docs):
        markdown_content = md(doc.page_content, heading_style="ATX")

        markdown_doc = Document(
            page_content=markdown_content,
            metadata={
                "source": str(path),
                "page_number": i + 1,
                "total_pages": len(docs),
                "file_type": "html",
                "content_format": "markdown",
            },
        )

        markdown_docs.append(markdown_doc)

    return markdown_docs


def txt_loader(path: Union[str, Path]) -> List[Document]:
    """
    Load a plain text earnings report.

    Args:
        path (Union[str, Path]): File path of the text document.

    Returns:
        List[Document]: One document with text content and metadata.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        sections = re.split(r"\n\s*\n", content.strip())

        markdown_docs = []

        for i, section in enumerate(sections):
            if section.strip():
                markdown_content = _convert_to_markdown(section.strip())

                markdown_doc = Document(
                    page_content=markdown_content,
                    metadata={
                        "source": str(path),
                        "page_number": i + 1,
                        "total_sections": len(sections),
                        "file_type": "txt",
                        "content_format": "markdown",
                    },
                )

                markdown_docs.append(markdown_doc)

        return (
            markdown_docs
            if markdown_docs
            else [
                Document(
                    page_content=_convert_to_markdown(content),
                    metadata={
                        "source": str(path),
                        "page_number": 1,
                        "total_sections": 1,
                        "file_type": "txt",
                        "content_format": "markdown",
                    },
                )
            ]
        )

    except Exception as e:
        return [
            Document(
                page_content=f"Error loading document: {str(e)}",
                metadata={
                    "source": str(path),
                    "page_number": 1,
                    "total_sections": 1,
                    "file_type": "txt",
                    "content_format": "markdown",
                    "error": str(e),
                },
            )
        ]


def _convert_to_markdown(text: str) -> str:
    """
    Convert plain text to markdown-like format.

    Args:
        text (str): Plain text content.

    Returns:
        str: Markdown-formatted content.
    """
    if not text:
        return ""

    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(
        r"^([A-Z][A-Z\s]+)$", r"# \1", text, flags=re.MULTILINE
    )
    text = re.sub(
        r"^([A-Z][a-z\s]+:)$", r"## \1", text, flags=re.MULTILINE
    ) 
    text = re.sub(
        r"^(\d+\.\s)", r"### \1", text, flags=re.MULTILINE
    ) 

    # Format tables (basic)
    lines = text.split("\n")
    formatted_lines = []

    for i, line in enumerate(lines):
        if re.match(r"^[\w\s]+\s{2,}[\w\s]+\s{2,}[\w\s]+", line):
            if i == 0 or not re.match(
                r"^[\w\s]+\s{2,}[\w\s]+\s{2,}[\w\s]+", lines[i - 1]
            ):
                formatted_lines.append("| " + " | ".join(line.split()) + " |")
                formatted_lines.append("|" + "---|" * (len(line.split()) - 1) + "---|")
            else:
                formatted_lines.append("| " + " | ".join(line.split()) + " |")
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split earnings report Documents into semantically organized and manageable segments.

    Splitting logic:
      1) Apply MarkdownHeaderTextSplitter to partition text by headers.
      2) Apply RecursiveCharacterTextSplitter for size-controlled splits.

    Args:
        documents (List[Document]): Documents with Markdown content.

    Returns:
        List[Document]: Child chunks with merged metadata (parent + original).
    """
    if not documents:
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )

    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, separators=["\n\n", "\n", ". ", " "]
    )

    final_chunks: List[Document] = []

    for doc in documents:
        parent_chunks = header_splitter.split_text(doc.page_content)

        for i, parent in enumerate(parent_chunks):
            child_docs = rec_splitter.create_documents(
                [parent.page_content],
                metadatas=[
                    {
                        **doc.metadata,
                        **parent.metadata, 
                        "section_index": i, 
                        "chunk_strategy": "header_then_recursive",
                    }
                ],
            )

            for j, child in enumerate(child_docs):
                child.metadata.update(
                    {
                        "chunk_index": j,
                        "total_chunks_in_section": len(child_docs),
                        "chunk_size": len(child.page_content),
                    }
                )
                final_chunks.append(child)

    return final_chunks
