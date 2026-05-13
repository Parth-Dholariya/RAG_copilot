from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.answering import answer_question
from app.config import settings
from app.embeddings import EmbeddingModel
from app.evaluate import evaluate
from app.ingest import ingest
from app.retrieval import retrieve_and_rerank
from app.schemas import Citation, EvaluationRequest, EvaluationResponse, IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.vector_store import VectorStore

app = FastAPI(
    title="Enterprise RAG Copilot",
    description="Document QA service with semantic retrieval, reranking, citations, and evaluation.",
    version="1.0.0",
)


def _store(index_dir: Path | None = None) -> VectorStore:
    return VectorStore(index_dir or settings.index_dir, EmbeddingModel(settings.embedding_model))


@app.get("/health")
def health() -> dict[str, str | bool]:
    index_ready = (settings.index_dir / "chunks.jsonl").exists() and (settings.index_dir / "vectors.npy").exists()
    return {"status": "ok", "index_ready": index_ready, "index_dir": str(settings.index_dir)}


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(request: IngestRequest) -> IngestResponse:
    try:
        documents, chunks, index_dir = ingest(Path(request.source_dir), Path(request.index_dir) if request.index_dir else None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(documents=documents, chunks=chunks, index_dir=str(index_dir))


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest) -> QueryResponse:
    store = _store()
    try:
        store.load()
        contexts = retrieve_and_rerank(store, request.question, request.top_k, request.rerank_top_n)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Query failed. Ingest documents first. {exc}") from exc

    answer = answer_question(request.question, contexts, settings)
    citations = [
        Citation(
            source=chunk.source,
            page=chunk.page,
            chunk_id=chunk.id,
            score=round(score, 4),
            excerpt=chunk.text[:500],
        )
        for chunk, score in contexts
    ]
    return QueryResponse(answer=answer, citations=citations)


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_endpoint(request: EvaluationRequest) -> EvaluationResponse:
    try:
        metrics = evaluate(Path(request.dataset_path), settings.index_dir, request.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return EvaluationResponse(**metrics)
