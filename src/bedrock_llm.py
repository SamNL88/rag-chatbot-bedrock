"""
Bedrock LLM interaction module.

This module formats prompts, calls AWS Bedrock (Claude), and parses model
responses. Retrieval context and model settings are controlled via config.
"""

import json
from typing import List, Dict, Any

import boto3
from botocore.config import Config as BotoConfig

from .logger import setup_logger
from .retriever import format_context
from .config import config

logger = setup_logger()

# Bedrock client
_bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=config.aws_region,
    config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"}),
)

# Prompt building


def build_prompt(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Build a prompt string combining retrieved context with the user question.
    """
    context = format_context(context_chunks)

    prompt = f"""
You are a helpful support assistant for the SmartHeat Pro thermostat.

You must ONLY use the information in the CONTEXT below to answer the user's question.
If the answer is not in the context, say you don't know and suggest that the user contact SmartHeat support.

CONTEXT:
{context}

QUESTION:
{question}

Answer in a concise, clear way, in 3 to 6 sentences at most.
If a specific document/source is important, mention it briefly.
"""
    return prompt.strip()

# Model invocation


def generate_answer(
    question: str,
    context_chunks: List[Dict[str, Any]],
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    Generate an LLM answer using AWS Bedrock (Claude 3).

    Parameters
    ----------
    question : str
        User question.
    context_chunks : list of dict
        Retrieved chunks to include in the prompt.
    max_tokens : int, optional
        Override for token limit; falls back to config.
    temperature : float, optional
        Override for temperature; falls back to config.

    Returns
    -------
    str
        The model-produced answer.
    """

    if not context_chunks:
        logger.warning(
            "No context chunks retrieved; answering without context.")

    prompt = build_prompt(question, context_chunks)

    # Fill defaults from central config if not provided
    if max_tokens is None:
        max_tokens = config.max_tokens
    if temperature is None:
        temperature = config.temperature

    logger.info(
        f"Calling Bedrock model: {config.model_id} in region: {config.aws_region}"
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": config.top_p,
    }

    try:
        response = _bedrock_runtime.invoke_model(
            modelId=config.model_id,
            body=json.dumps(body),
        )
    except Exception as e:
        logger.error(f"Error calling Bedrock: {e}")
        raise

    # Bedrock response body: streaming-like object
    raw_body = response.get("body")
    if hasattr(raw_body, "read"):
        raw_body = raw_body.read()

    payload = json.loads(raw_body)

    # Claude-specific response format
    try:
        answer = payload["content"][0]["text"]
    except Exception as e:
        logger.error(f"Unexpected Bedrock response format: {payload}")
        raise e

    logger.info("Received response from Bedrock.")
    return answer.strip()
