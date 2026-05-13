# Enterprise RAG Copilot Instructions

This repository contains a FastAPI document QA service for academic and organizational documents.

## Project Status

- Project scaffolded with API, ingestion, retrieval, answer generation, evaluation, Docker, and VS Code tasks.
- Sample Markdown documents live in `data/sample_docs`.
- Evaluation examples live in `eval/qa_pairs.jsonl`.
- Runtime indexes are generated under `storage/index` and are ignored by Git.

## Development Workflow

1. Install required dependencies with `pip install -r requirements.txt`.
2. Optionally install stronger retrieval and hosted LLM integrations with `pip install -r requirements-optional.txt`.
3. Ingest documents with `python -m app.ingest --source data/sample_docs --index storage/index`.
4. Run the API with `uvicorn app.main:app --reload`.
5. Evaluate retrieval and answer grounding with `python -m app.evaluate --dataset eval/qa_pairs.jsonl --index storage/index --top-k 5`.

## Architecture Notes

- `app/document_loader.py` loads PDF, Markdown, and text documents.
- `app/chunking.py` performs paragraph-aware semantic chunking with overlap.
- `app/embeddings.py` uses Sentence Transformers when installed and falls back to deterministic hash embeddings.
- `app/vector_store.py` uses FAISS when installed and falls back to NumPy cosine search.
- `app/retrieval.py` reranks retrieved chunks with semantic and lexical signals.
- `app/answering.py` supports extractive answers by default and optional OpenAI or Gemini generation.
- `app/evaluate.py` reports Recall@K, MRR, faithfulness proxy, and answer relevance proxy.

Keep future changes focused on grounded document QA, citations, evaluation quality, and enterprise-search ergonomics.
