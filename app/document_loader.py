from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RawDocument:
    text: str
    source: str
    page: int | None = None


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def load_documents(source_dir: Path) -> list[RawDocument]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Document directory does not exist: {source_dir}")

    documents: list[RawDocument] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path, source_dir))
        else:
            documents.append(
                RawDocument(
                    text=path.read_text(encoding="utf-8", errors="ignore"),
                    source=str(path.relative_to(source_dir)),
                )
            )
    return [doc for doc in documents if doc.text.strip()]


def _load_pdf(path: Path, root: Path) -> list[RawDocument]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install pypdf to ingest PDF documents.") from exc

    reader = PdfReader(str(path))
    docs: list[RawDocument] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(
                RawDocument(
                    text=text,
                    source=str(path.relative_to(root)),
                    page=page_number,
                )
            )
    return docs
