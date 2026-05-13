# Enterprise RAG Copilot

This is a small but complete RAG service for asking questions over PDFs, university notes, research summaries, and policy documents. I built it to look like the kind of document QA workflow used in enterprise search: ingest files, retrieve relevant passages, rerank them, answer with citations, and evaluate whether retrieval is actually working.

## What It Does

- PDF, Markdown, and text ingestion
- Paragraph-aware chunking with overlap
- Vector search with FAISS when installed, plus a NumPy fallback for easy demos
- Sentence Transformers when available, plus a deterministic local embedding fallback
- Lightweight reranking that blends semantic similarity and keyword overlap
- Citation-backed answers with source, page, chunk id, score, and excerpt
- Optional OpenAI or Gemini generation
- Evaluation for Recall@K, MRR, answer relevance, and a simple faithfulness proxy

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.ingest --source data/sample_docs --index storage/index
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. If Windows blocks port `8000`, use another port:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8010
```

## Ask A Question

```powershell
curl -X POST http://127.0.0.1:8000/query `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"How does the university policy handle data privacy?\",\"top_k\":5}"
```

## Optional LLM Providers

The default answerer is extractive, so the app can run without an API key. For stronger natural-language responses, install the optional integrations:

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

The sample evaluation file is intentionally tiny. Add your own questions and expected source files to `eval/qa_pairs.jsonl`, then run:

```powershell
python -m app.evaluate --dataset eval/qa_pairs.jsonl --index storage/index --top-k 5
```

Metrics include:

- `recall_at_k`: Whether the expected source appears in retrieved contexts
- `mrr`: Rank quality for expected supporting documents
- `faithfulness`: Rough support check for answer sentences against retrieved context
- `answer_relevance`: Token overlap between question and answer

## API Endpoints

- `GET /health` - service and index status
- `POST /ingest` - ingest documents from a server-side folder
- `POST /query` - retrieve contexts and generate a grounded answer
- `POST /evaluate` - run an evaluation dataset

## Project Layout

```text
app/
  answering.py        grounded answer generation
  chunking.py         document splitting
  document_loader.py  PDF/Markdown/text loading
  embeddings.py       embedding model wrapper and local fallback
  evaluate.py         offline RAG metrics
  main.py             FastAPI routes
  retrieval.py        vector retrieval and reranking
  vector_store.py     persisted vector index
```

## Resume Bullet

Built an Enterprise RAG-based Copilot for querying academic and organizational documents using semantic chunking, vector search, reranking, citation-grounded answer generation, and Recall@K/MRR/faithfulness evaluation to reduce hallucinations and improve response reliability.
