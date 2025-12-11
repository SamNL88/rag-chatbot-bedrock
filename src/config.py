"""
Central configuration module for the RAG chatbot.

All configurable parameters (retrieval, LLM settings, ingestion paths, and
chunking rules) are defined here.
"""


from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Config:
    """
    Application-wide configuration values.

    Defaults are defined here but can be overridden using environment
    variables to support development, testing, and deployment setups.
    """

    # Retrieval / RAG
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))

    # LLM / Bedrock
    aws_region: str = os.getenv("AWS_REGION", "eu-central-1")
    model_id: str = os.getenv(
        "BEDROCK_MODEL_ID",
        "anthropic.claude-3-5-sonnet-20240620-v1:0",
    )
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))

    # Ingestion / preprocessing
    # Relative folder names under project root
    docs_subdir: str = os.getenv("DOCS_SUBDIR", "docs")
    data_subdir: str = os.getenv("DATA_SUBDIR", "data")

    # Chunking parameters
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Output filenames for embeddings and metadata
    embeddings_filename: str = os.getenv(
        "EMBEDDINGS_FILENAME", "chunks_embeddings.npy"
    )
    meta_filename: str = os.getenv("META_FILENAME", "chunks_meta.npy")


# Single shared instance imported everywhere
config = Config()
