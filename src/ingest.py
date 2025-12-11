"""
Ingestion pipeline for preparing documents for retrieval.

This script:
1. Loads raw text files from the docs directory.
2. Splits them into overlapping text chunks.
3. Embeds all chunks using SentenceTransformers.
4. Saves embeddings and metadata to the data directory.

These outputs are later used by the retriever for similarity search.
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .embeddings import embed_texts
from .logger import setup_logger
from .config import config


logger = setup_logger()


# Setting the paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / config.docs_subdir
DATA_DIR = ROOT_DIR / config.data_subdir
DATA_DIR.mkdir(exist_ok=True)


def load_documents() -> List[Dict[str, Any]]:
    """
    Load all .txt documents from the docs directory.

    Returns
    -------
    List[Dict[str, Any]]
        Each item contains {"source": filename, "text": file_contents}.
    """

    docs: List[Dict[str, Any]] = []
    for path in DOCS_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "text": text})

    return docs


def chunk_documents(
        documents: List[Dict[str, Any]],
        chunk_size: int = config.chunk_size,
        chunk_overlap: int = config.chunk_overlap,
) -> List[Dict[str, Any]]:
    """
    Split documents into overlapping text chunks using LangChain's splitter.

    Parameters
    ----------
    documents : List[Dict[str, Any]]
        Input documents from load_documents().
    chunk_size : int
        Maximum length of each chunk.
    chunk_overlap : int
        Overlap between consecutive chunks.

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries: {"id", "source", "text"}.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for doc in documents:
        source = doc["source"]
        text = doc["text"]

        chunks = splitter.split_text(text)
        logger.info(f"Chunked {source} into {len(chunks)} chunks")

        for ch in chunks:
            ch = ch.strip()
            if ch:
                all_chunks.append({
                    "id": chunk_id,
                    "source": source,
                    "text": ch
                })
                chunk_id += 1

    return all_chunks


def main() -> None:
    """Run the ingestion pipeline and save embeddings + metadata to disk."""

    logger.info(f"Loading documents from: {DOCS_DIR}")
    documents = load_documents()
    logger.info(f"Loaded {len(documents)} documents")

    logger.info("Splitting using RecursiveCharacterTextSplitter...")
    chunks = chunk_documents(documents)
    logger.info(f"Total chunks created: {len(chunks)}")

    texts = [c["text"] for c in chunks]

    logger.info("Embedding all chunks...")
    embeddings = embed_texts(texts)
    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings + metadata
    embeddings_path = DATA_DIR / config.embeddings_filename
    meta_path = DATA_DIR / config.meta_filename

    np.save(embeddings_path, embeddings)

    # Structured Numpy array for metadata storage
    meta_dtype = np.dtype([
        ("id", np.int32),
        ("source", "U128"),
        ("text", "U4000"),
    ])

    meta = np.empty(len(chunks), dtype=meta_dtype)
    for i, c in enumerate(chunks):
        meta[i]["id"] = c["id"]
        meta[i]["source"] = c["source"]
        meta[i]["text"] = c["text"]

    np.save(meta_path, meta)

    logger.success(f"Saved embeddings -> {embeddings_path}")
    logger.success(f"Saved metadata   -> {meta_path}")
    logger.success("Ingestion completed successfully!")


if __name__ == "__main__":
    main()
