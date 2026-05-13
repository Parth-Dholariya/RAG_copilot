from dataclasses import dataclass
import re
from uuid import uuid5, NAMESPACE_URL

from app.document_loader import RawDocument


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int | None


def chunk_documents(docs: list[RawDocument], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for text in _semantic_windows(doc.text, chunk_size, overlap):
            chunk_id = str(uuid5(NAMESPACE_URL, f"{doc.source}:{doc.page}:{text[:120]}"))
            chunks.append(Chunk(id=chunk_id, text=text, source=doc.source, page=doc.page))
    return chunks


def _semantic_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    windows: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                windows.append(current.strip())
                current = ""
            windows.extend(_split_long_text(paragraph, chunk_size, overlap))
            continue
        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                windows.append(current.strip())
            current = paragraph
    if current:
        windows.append(current.strip())

    if overlap <= 0 or len(windows) < 2:
        return windows

    overlapped: list[str] = []
    previous_tail = ""
    for window in windows:
        merged = f"{previous_tail}\n\n{window}".strip() if previous_tail else window
        overlapped.append(merged)
        previous_tail = _sentence_tail(window, overlap)
    return overlapped


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    if len(chunks) == 1:
        return chunks

    output: list[str] = []
    tail = ""
    for chunk in chunks:
        output.append(f"{tail} {chunk}".strip() if tail else chunk)
        tail = chunk[-overlap:] if overlap > 0 else ""
    return output


def _sentence_tail(text: str, max_chars: int) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    tail = ""
    for sentence in reversed(sentences):
        candidate = f"{sentence} {tail}".strip()
        if len(candidate) > max_chars and tail:
            break
        tail = candidate
    return tail[-max_chars:] if len(tail) > max_chars else tail
