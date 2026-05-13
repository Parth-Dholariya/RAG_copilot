import argparse
from pathlib import Path

from app.chunking import chunk_documents
from app.config import settings
from app.document_loader import load_documents
from app.embeddings import EmbeddingModel
from app.vector_store import VectorStore


def ingest(source: Path, index_dir: Path | None = None) -> tuple[int, int, Path]:
    target_index = index_dir or settings.index_dir
    docs = load_documents(source)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    store = VectorStore(target_index, EmbeddingModel(settings.embedding_model))
    store.build(chunks)
    return len(docs), len(chunks), target_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector index.")
    parser.add_argument("--source", type=Path, default=settings.document_dir)
    parser.add_argument("--index", type=Path, default=settings.index_dir)
    args = parser.parse_args()

    documents, chunks, index_dir = ingest(args.source, args.index)
    print(f"Ingested {documents} documents into {chunks} chunks at {index_dir}")


if __name__ == "__main__":
    main()
