import json
from pathlib import Path
from typing import Any

import numpy as np

from app.chunking import Chunk
from app.embeddings import EmbeddingModel


class VectorStore:
    def __init__(self, index_dir: Path, embedding_model: EmbeddingModel):
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self._faiss_index: Any | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = chunks
        self.vectors = self.embedding_model.encode([chunk.text for chunk in chunks])
        self._try_build_faiss()
        self._persist()

    def load(self) -> None:
        metadata_path = self.index_dir / "chunks.jsonl"
        vectors_path = self.index_dir / "vectors.npy"
        if not metadata_path.exists() or not vectors_path.exists():
            raise FileNotFoundError(f"No index found in {self.index_dir}")

        self.chunks = []
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            self.chunks.append(Chunk(**item))
        self.vectors = np.load(vectors_path).astype(np.float32)
        self._try_load_faiss()

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self.vectors is None:
            self.load()
        assert self.vectors is not None
        query_vector = self.embedding_model.encode([query]).astype(np.float32)

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query_vector, min(top_k, len(self.chunks)))
            return [
                (self.chunks[int(index)], float(score))
                for index, score in zip(indices[0], scores[0], strict=False)
                if int(index) >= 0
            ]

        scores = np.dot(self.vectors, query_vector[0])
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[int(index)], float(scores[int(index)])) for index in top_indices]

    def _persist(self) -> None:
        assert self.vectors is not None
        metadata = "\n".join(json.dumps(chunk.__dict__) for chunk in self.chunks)
        (self.index_dir / "chunks.jsonl").write_text(metadata + "\n", encoding="utf-8")
        np.save(self.index_dir / "vectors.npy", self.vectors)
        if self._faiss_index is not None:
            try:
                import faiss

                faiss.write_index(self._faiss_index, str(self.index_dir / "faiss.index"))
            except Exception:
                pass

    def _try_build_faiss(self) -> None:
        self._faiss_index = None
        try:
            import faiss

            assert self.vectors is not None
            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors.astype(np.float32))
            self._faiss_index = index
        except Exception:
            self._faiss_index = None

    def _try_load_faiss(self) -> None:
        self._faiss_index = None
        faiss_path = self.index_dir / "faiss.index"
        if not faiss_path.exists():
            return
        try:
            import faiss

            self._faiss_index = faiss.read_index(str(faiss_path))
        except Exception:
            self._faiss_index = None
