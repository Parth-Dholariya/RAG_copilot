from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    source_dir: str = Field(..., description="Server-side folder containing PDF, Markdown, or text files.")
    index_dir: str | None = Field(default=None, description="Optional output index directory.")


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    index_dir: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    rerank_top_n: int = 5


class Citation(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]


class EvaluationRequest(BaseModel):
    dataset_path: str
    top_k: int = 5


class EvaluationResponse(BaseModel):
    recall_at_k: float
    mrr: float
    faithfulness: float
    answer_relevance: float
    examples: int
