"""
Simple vector-based retriever for RAG.

This module:
1. Loads precomputed chunk embeddings + metadata.
2. Embeds user queries.
3. Computes cosine similarity via dot product.
4. Returns the top-k most relevant chunks.
"""


from pathlib import Path
from typing import List, Dict, Any

import numpy as np

from .embeddings import embed_texts
from .logger import setup_logger
from .config import config


logger = setup_logger()

# Paths to stored embeddings and metadata
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / config.data_subdir

EMBEDDINGS_PATH = DATA_DIR / config.embeddings_filename
META_PATH = DATA_DIR / config.meta_filename

# Globals for lazy-loaded index
_CHUNK_EMBEDDINGS: np.ndarray | None = None
_CHUNK_META: np.ndarray | None = None


def _load_index() -> None:
    """
    Load embeddings and metadata into memory (only once).

    Uses lazy loading so the index is loaded on first retrieval call,
    not during import time.
    """

    global _CHUNK_EMBEDDINGS, _CHUNK_META

    if _CHUNK_EMBEDDINGS is not None and _CHUNK_META is not None:
        return

    logger.info(f"Loading embeddings from {EMBEDDINGS_PATH}")
    logger.info(f"Loading metadata   from {META_PATH}")

    _CHUNK_EMBEDDINGS = np.load(EMBEDDINGS_PATH)
    _CHUNK_META = np.load(META_PATH)

    logger.info(f"Loaded embeddings with shape: {_CHUNK_EMBEDDINGS.shape}")
    logger.info(f"Loaded meta with shape: {_CHUNK_META.shape}")


def retrieve_relevant_chunks(
    query: str,
    top_k: int = config.top_k,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most similar document chunks to a query.

    Parameters
    ----------
    query : str
        User query to embed and compare.
    top_k : int
        Number of chunks to return. Defaults to config.top_k.

    Returns
    -------
    List[Dict[str, Any]]
        Each entry contains {"id", "source", "text", "score"}.
    """

    _load_index()
    assert _CHUNK_EMBEDDINGS is not None
    assert _CHUNK_META is not None

    # Embed query → (D,)
    query_emb = embed_texts([query])[0]
    query_emb = query_emb.astype(np.float32)

    # Embeddings are normalized => cosine similarity = dot product
    scores = _CHUNK_EMBEDDINGS @ query_emb  # shape (N,)

    # Limit top_k to available chunks
    top_k = min(top_k, scores.shape[0])
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        meta_row = _CHUNK_META[idx]
        results.append(
            {
                "id": int(meta_row["id"]),
                "source": str(meta_row["source"]),
                "text": str(meta_row["text"]),
                "score": float(scores[idx]),
            }
        )

    logger.info(
        f"Retrieved top {top_k} chunks for query: {query!r}"
    )
    return results


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a single context string for prompts.
    Includes source information and similarity score.
    """
    parts: List[str] = []
    for c in chunks:
        header = f"[Source: {c['source']} | Score: {c['score']:.3f}]"
        parts.append(header)
        parts.append(c["text"])
        parts.append("")
    return "\n".join(parts)
