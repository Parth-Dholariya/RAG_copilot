import argparse
import json
import re
from pathlib import Path

from app.answering import answer_question
from app.config import settings
from app.embeddings import EmbeddingModel
from app.retrieval import retrieve_and_rerank
from app.vector_store import VectorStore


def evaluate(dataset_path: Path, index_dir: Path, top_k: int) -> dict[str, float | int]:
    store = VectorStore(index_dir, EmbeddingModel(settings.embedding_model))
    store.load()
    examples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    recall_hits = 0
    reciprocal_ranks: list[float] = []
    faithfulness_scores: list[float] = []
    relevance_scores: list[float] = []

    for example in examples:
        question = example["question"]
        expected_sources = set(example.get("expected_sources", []))
        contexts = retrieve_and_rerank(store, question, top_k, top_k)
        sources = [chunk.source for chunk, _ in contexts]
        answer = answer_question(question, contexts, settings)

        hit_rank = next((idx for idx, source in enumerate(sources, start=1) if source in expected_sources), None)
        if hit_rank is not None:
            recall_hits += 1
            reciprocal_ranks.append(1 / hit_rank)
        else:
            reciprocal_ranks.append(0.0)

        context_text = " ".join(chunk.text for chunk, _ in contexts)
        faithfulness_scores.append(_faithfulness(answer, context_text))
        relevance_scores.append(_overlap(question, answer))

    total = max(len(examples), 1)
    return {
        "recall_at_k": recall_hits / total,
        "mrr": sum(reciprocal_ranks) / total,
        "faithfulness": sum(faithfulness_scores) / total,
        "answer_relevance": sum(relevance_scores) / total,
        "examples": len(examples),
    }


def _faithfulness(answer: str, context: str) -> float:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0.0
    supported = sum(1 for sentence in sentences if _overlap(sentence, context) >= 0.35)
    return supported / len(sentences)


def _overlap(left: str, right: str) -> float:
    left_terms = set(re.findall(r"[a-zA-Z0-9]+", left.lower()))
    right_terms = set(re.findall(r"[a-zA-Z0-9]+", right.lower()))
    if not left_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and grounded answer quality.")
    parser.add_argument("--dataset", type=Path, default=Path("eval/qa_pairs.jsonl"))
    parser.add_argument("--index", type=Path, default=settings.index_dir)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    args = parser.parse_args()

    metrics = evaluate(args.dataset, args.index, args.top_k)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
