import re

from app.chunking import Chunk
from app.vector_store import VectorStore


def retrieve_and_rerank(store: VectorStore, question: str, top_k: int, rerank_top_n: int) -> list[tuple[Chunk, float]]:
    candidates = store.search(question, max(top_k * 3, rerank_top_n))
    question_terms = _terms(question)
    reranked: list[tuple[Chunk, float]] = []
    for chunk, semantic_score in candidates:
        chunk_terms = _terms(chunk.text)
        lexical_score = len(question_terms & chunk_terms) / max(len(question_terms), 1)
        combined = (0.75 * semantic_score) + (0.25 * lexical_score)
        reranked.append((chunk, combined))
    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked[:rerank_top_n]


def _terms(text: str) -> set[str]:
    stopwords = {"the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "is", "are", "be"}
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in stopwords}
