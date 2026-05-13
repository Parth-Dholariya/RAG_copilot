import re

from app.chunking import Chunk
from app.config import Settings


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
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        for sentence in sentences:
            terms = _content_terms(sentence)
            if not sentence.strip() or not terms:
                continue
            score = len(question_terms & terms) / len(question_terms or {"_"})
            scored_sentences.append((score, sentence.strip(), citation_number))

    selected = [item for item in sorted(scored_sentences, reverse=True) if item[0] >= 0.3][:3]
    if not selected:
        selected = sorted(scored_sentences, reverse=True)[:1]
    if not selected:
        return "I could not find enough supporting context in the indexed documents to answer that question."
    return " ".join(f"{sentence} [{citation}]" for _, sentence, citation in selected)


def _content_terms(text: str) -> set[str]:
    stopwords = {
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
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if token not in stopwords}
