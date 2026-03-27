# Asistente Jurídico Mexicano

Backend service for a legal assistant focused on Mexican law, built with Python, FastAPI, PostgreSQL, and vector search.

This project is designed to ingest legal documents, structure them into retrievable legal units, and answer user queries through a hybrid retrieval pipeline that combines lexical search, vector similarity, and legal reranking.

---

## Overview

The main goal of this project is to provide grounded legal answers based on structured legal sources rather than generic text generation.

The system includes:

- legal document ingestion pipelines
- parsing and normalization of legal texts
- chunking into legal units such as articles, sections, fractions, and subsections
- hybrid retrieval (full-text + vector)
- legal reranking logic
- citation-aware answer generation
- API endpoints for ingestion and query workflows

This repository represents the backend and retrieval layer of the project.

---

## Main Features

- Ingestion of Mexican legal documents
- Parsing and structuring of legal texts
- Hybrid retrieval over legal sources
- Legal reranking to improve relevance
- FastAPI endpoints for querying and ingestion
- PostgreSQL + pgvector support
- Traceable, evidence-based responses

---

## Tech Stack

- **Python**
- **FastAPI**
- **PostgreSQL**
- **pgvector**
- **SQLAlchemy / psycopg**
- **OpenAI API**
- **Alembic**

---

## Project Structure

```text
app/
  api/          # API routes
  audit/        # audit / traceability logic
  core/         # configuration, logging, security
  ingestion/    # ETL, parsing and ingestion pipelines
  services/     # retrieval, reranking, citations, RAG services
  main.py       # FastAPI entry point

scripts/
  init_db.sql
  run_etl.py
  seed_demo.py

tests/
README.md
requirements.txt
alembic.ini
.env.example

Core Components

1. Ingestion

The ingestion layer parses legal sources and transforms them into structured database records.

Main responsibilities:

extract raw legal text
normalize structure
split content into legal units
assign metadata such as article, section, fraction, or subsection
persist documents, chunks, and embeddings
2. Retrieval

The retrieval layer combines:

lexical search
vector similarity
legal-specific reranking

This is intended to improve precision over standard keyword-only search.

3. RAG Service

The RAG service orchestrates:

query understanding
retrieval
reranking
evidence preparation
response generation
citations
4. API Layer

FastAPI routes expose ingestion and query capabilities for external clients.

Why This Project Is Different

This is not a generic chatbot wrapper.

The differentiating value lies in:

structured legal ingestion
hybrid retrieval over legal sources
domain-specific reranking
evidence-based response generation
legal traceability and citations

The project is designed to reduce hallucinations and improve retrieval relevance in the legal domain.

Setup
1. Clone the repository

git clone https://github.com/YOUR_USERNAME/asistente-juridico.git
cd asistente-juridico

2. Create virtual environment

python -m venv .venv
source .venv/bin/activate

On Windows:
.venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Configure environment variables

Create a .env file based on .env.example.

Example:

DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/asistente_juridico
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_MAX_RETRIES=3

5. Run the app

uvicorn app.main:app --reload

API

Main endpoints may include:

POST /ingest
POST /ask
GET /health

Adjust this section to match your current routes.

Notes

This public version does not include private datasets, credentials, or sensitive environment configuration.
Some project-specific data sources or internal prompts may not be included in the public repository.
The repository is intended to demonstrate system design, ingestion, retrieval, and legal reranking logic.
Future Improvements
stronger evaluation framework
better retrieval diagnostics
more robust citation formatting
support for additional legal corpora
deployment hardening
public demo environment

---
##  Author
## Gaddiel Ramos
---