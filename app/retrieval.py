import re

from app.chunking import Chunk
from app.vector_store import VectorStore


SEMANTIC_WEIGHT = 0.75
LEXICAL_WEIGHT = 0.25
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "for",
    "on",
    "with",
    "is",
    "are",
    "be",
}


def retrieve_and_rerank(store: VectorStore, question: str, top_k: int, rerank_top_n: int) -> list[tuple[Chunk, float]]:
    candidates = store.search(question, max(top_k * 3, rerank_top_n))
    question_terms = _terms(question)

    reranked: list[tuple[Chunk, float]] = []
    for chunk, semantic_score in candidates:
        lexical_score = _lexical_overlap(question_terms, _terms(chunk.text))
        combined = (SEMANTIC_WEIGHT * semantic_score) + (LEXICAL_WEIGHT * lexical_score)
        reranked.append((chunk, combined))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked[:rerank_top_n]


def _terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in STOPWORDS}


def _lexical_overlap(query_terms: set[str], chunk_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & chunk_terms) / len(query_terms)
