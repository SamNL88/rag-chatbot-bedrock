# SmartHeat Pro – RAG Chatbot (AWS Bedrock)

A lightweight Retrieval-Augmented Generation (RAG) chatbot built on  
AWS Bedrock (Claude 3), SentenceTransformer embeddings, and a simple vector search engine.  
The project is structured as a clean, modular demo showing how to build a practical RAG system end-to-end.

**Note on the product:**  
This chatbot is designed around an entirely **imaginary thermostat product** called **SmartHeat Pro**.  
The product, brand, and documentation used in this project are fictional and created solely for educational and demonstration purposes.

**What SmartHeat Pro represents:**  
A simple home thermostat device with features such as Wi-Fi connectivity, scheduling, mobile app controls, and multi-zone support.  
The included documents simulate typical customer-facing materials such as:

- General FAQs
- Getting started guides
- Integration notes
- Troubleshooting steps
- Usage instructions

All content is generic and invented to avoid copyright issues.

**What users might ask the chatbot:**

- “How do I connect SmartHeat Pro to Wi-Fi?”
- “Can multiple users control the same thermostat?”
- “How accurate is the temperature sensor?”
- “How do firmware updates work?”

These example questions and answers are also fictional and created solely to demonstrate how RAG retrieves relevant chunks and builds contextual answers.

---

## Quick Start

### 1. Create and activate virtual environment

```bash
python -m venv .venv

.venv\Scripts\activate              # Windows

source .venv/bin/activate           # macOS / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure AWS

```bash
aws configure
```

### 4. Ingest documents

```bash
python -m src.ingest
```

### 5. Run the chatbot

```bash
python main.py
```

---

## Project Structure

```text
src/
 ├── chatbot_cli.py      # Chatbot CLI interface
 ├── config.py           # Central configuration module
 ├── logger.py           # Logging setup (Loguru)
 ├── embeddings.py       # Embedding utilities (SentenceTransformer)
 ├── retriever.py        # Vector search using cosine similarity
 ├── bedrock_llm.py      # AWS Bedrock model interface
 └── ingest.py           # Document preprocessing and embedding generation

docs/                    # Input text files (fictional product documentation)
data/                    # Generated embeddings + metadata after ingestion

.gitignore               # Files/folders excluded from version control
README.md                # Project documentation
requirements.txt         # Python dependency list
main.py                  # External entrypoint that launches the CLI

```

---

## Configuration

All settings (chunk sizes, model ID, region, top_k, etc.) are defined in:
`src/config.py`

---

## How It Works

1. Ingestion

- Loads documents
- Splits into overlapping chunks (Recursive chunking)
- Generates embeddings
- Saves them to data/

2. Retrieval

- Embeds the user query
- Computes cosine similarity
- Returns top-k chunks

3. Generation

- Builds a contextual prompt
- Sends it to AWS Bedrock (Claude)
- Returns the model answer

---

## AI Disclaimer

All product documentation used in this project (FAQs, guides, integration notes, troubleshooting text, and usage instructions) is entirely fictional and was generated with the assistance of AI tools (ChatGPT).  
This was done intentionally to avoid copyright issues and ensure that no real proprietary product information is used.

---

## Dependencies

This project relies on boto3, numpy, sentence-transformers, and loguru. Full version-pinned list available in requirements.txt.

---

## Summary

This project demonstrates a clean and modular Retrieval-Augmented Generation (RAG) chatbot built on AWS Bedrock (Claude 3).  
It includes document ingestion, embedding generation, vector retrieval, and contextual LLM responses.  
The entire pipeline is lightweight, transparent, and easy to extend — making it suitable for learning, experimentation, and building production-grade assistants.

---

## Possible Future Upgrades

Possible next steps for this project include adding persistent storage layers such as Amazon S3 for saving documents, embeddings or index snapshots, and long-term conversation history or summaries, while using DynamoDB to store structured per-user data, enabling both short-term and long-term memory tied to unique user IDs. The system could also be extended with multi-agent components (e.g., separate reasoning, planning, or retrieval agents), as well as enhanced retrieval quality through contextual retrieval techniques, hybrid search, or improved reranking. Together, these upgrades would transform the simple CLI chatbot into a more intelligent, personalized, and production-ready assistant.
