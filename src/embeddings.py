"""

Embedding utilities using SentenceTransformers.

"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


# initial implementation : Just MiniLM

_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Compute L2-normalized embeddings for a list of text strings.

    Parameters
    ----------
    texts : List[str]
        Text inputs to embed.

    Returns
    -------
    np.ndarray
        Array of shape (n_texts, embedding_dim) with float32 embeddings.
    """

    embeddings = _model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings, dtype=np.float32)
