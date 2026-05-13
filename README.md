# Enterprise RAG Copilot

A FastAPI-based copilot for querying research papers, university notes, policies, and organizational documents with retrieval-augmented generation, citations, reranking, and evaluation metrics.

## Features

- PDF, Markdown, and text ingestion
- Semantic chunking with configurable overlap
- FAISS vector search when available, with a NumPy fallback index
- Sentence Transformers embeddings when available, with a deterministic local fallback
- Hybrid reranking using semantic score and lexical overlap
- Citation-grounded answers with source file, page, and chunk metadata
- Optional OpenAI or Gemini answer generation
- Built-in evaluation for Recall@K, MRR, faithfulness proxy, and answer relevance proxy
- Dockerized API

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.ingest --source data/sample_docs --index storage/index
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Ask A Question

```powershell
curl -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"How does the university policy handle data privacy?\",\"top_k\":5}"
```

## Optional LLM Providers

The project runs with an extractive grounded answerer by default. To enable hosted generation:

```powershell
pip install -r requirements-optional.txt
```

```powershell
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="..."
```

or:

```powershell
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="..."
```

## Evaluation

Edit `eval/qa_pairs.jsonl` with expected source files and run:

```powershell
python -m app.evaluate --dataset eval/qa_pairs.jsonl --index storage/index --top-k 5
```

Metrics include:

- `recall_at_k`: Whether the expected source appears in retrieved contexts
- `mrr`: Rank quality for expected supporting documents
- `faithfulness`: Fraction of answer sentences supported by retrieved contexts
- `answer_relevance`: Token overlap between question and answer

## API Endpoints

- `GET /health` - service and index status
- `POST /ingest` - ingest documents from a server-side folder
- `POST /query` - retrieve contexts and generate a grounded answer
- `POST /evaluate` - run an evaluation dataset

## Resume Bullet

Built an Enterprise RAG-based Copilot for querying academic and organizational documents using semantic chunking, vector search, reranking, citation-grounded answer generation, and Recall@K/MRR/faithfulness evaluation to reduce hallucinations and improve response reliability.
