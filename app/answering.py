import re

from app.chunking import Chunk
from app.config import Settings

ANSWER_SENTENCE_THRESHOLD = 0.3
MAX_EXTRACTIVE_SENTENCES = 3
NO_EVIDENCE_MESSAGE = "I could not find enough supporting context in the indexed documents to answer that question."

CONTENT_STOPWORDS = {
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
    "should",
    "what",
    "which",
    "how",
    "into",
    "not",
    "must",
    "may",
}


def answer_question(question: str, contexts: list[tuple[Chunk, float]], settings: Settings) -> str:
    provider = settings.llm_provider.lower()
    if provider == "openai" and settings.openai_api_key:
        return _openai_answer(question, contexts, settings)
    if provider == "gemini" and settings.gemini_api_key:
        return _gemini_answer(question, contexts, settings)
    return _extractive_answer(question, contexts)


def _prompt(question: str, contexts: list[tuple[Chunk, float]]) -> str:
    context_text = "\n\n".join(
        f"[{index}] Source: {chunk.source}, page: {chunk.page or 'n/a'}, chunk: {chunk.id}\n{chunk.text}"
        for index, (chunk, _) in enumerate(contexts, start=1)
    )
    return (
        "Answer the question using only the provided context. "
        "If the context is insufficient, say what is missing. "
        "Include bracketed citation numbers for every factual claim.\n\n"
        f"Question: {question}\n\nContext:\n{context_text}\n\nAnswer:"
    )


def _openai_answer(question: str, contexts: list[tuple[Chunk, float]], settings: Settings) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": _prompt(question, contexts)}],
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def _gemini_answer(question: str, contexts: list[tuple[Chunk, float]], settings: Settings) -> str:
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model)
    response = model.generate_content(_prompt(question, contexts))
    return response.text


def _extractive_answer(question: str, contexts: list[tuple[Chunk, float]]) -> str:
    question_terms = _content_terms(question)
    scored_sentences: list[tuple[float, str, int]] = []

    for citation_number, (chunk, _) in enumerate(contexts, start=1):
        scored_sentences.extend(_score_sentences(chunk.text, question_terms, citation_number))

    selected = [
        item
        for item in sorted(scored_sentences, reverse=True)
        if item[0] >= ANSWER_SENTENCE_THRESHOLD
    ][:MAX_EXTRACTIVE_SENTENCES]

    if not selected:
        selected = sorted(scored_sentences, reverse=True)[:1]
    if not selected:
        return NO_EVIDENCE_MESSAGE
    return " ".join(f"{sentence} [{citation}]" for _, sentence, citation in selected)


def _score_sentences(text: str, question_terms: set[str], citation_number: int) -> list[tuple[float, str, int]]:
    scored: list[tuple[float, str, int]] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        clean_sentence = sentence.strip()
        terms = _content_terms(clean_sentence)
        if not clean_sentence or not terms:
            continue
        score = len(question_terms & terms) / len(question_terms or {"_"})
        scored.append((score, clean_sentence, citation_number))
    return scored


def _content_terms(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in CONTENT_STOPWORDS}
