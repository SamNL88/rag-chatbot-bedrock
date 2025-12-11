"""
Command-line interface for the SmartHeat Pro RAG chatbot.

This module handles:
1. User input loop
2. Retrieval of relevant chunks
3. LLM answer generation via Bedrock
4. Displaying formatted responses
"""

from .logger import setup_logger
from .retriever import retrieve_relevant_chunks
from .bedrock_llm import generate_answer
from .config import config


logger = setup_logger()


def main() -> None:
    """
    Run the interactive chatbot loop.
    """

    logger.info("SmartHeat Pro - RAG Chatbot (AWS Bedrock)")
    logger.info("Type your question, or 'exit' to quit.\n")

    while True:
        # User input
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            logger.info("Exiting chatbot.")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            logger.info("Goodbye!")
            break

        # 1) Retrieve relevant chunks
        chunks = retrieve_relevant_chunks(question, top_k=config.top_k)

        if not chunks:
            logger.warning(
                "No relevant chunks found; answering without context.")
        else:
            top_sources = {c['source'] for c in chunks}
            logger.info(f"Top sources used: {', '.join(top_sources)}")

        # 2) Call Bedrock LLM
        try:
            answer = generate_answer(question, chunks)
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            print("Bot: Sorry, something went wrong while generating the answer.")
            continue

        # 3) Show answer
        print("\nBot:", answer)
        print("-" * 80)


if __name__ == "__main__":
    main()
