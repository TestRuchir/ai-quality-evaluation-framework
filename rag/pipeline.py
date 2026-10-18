"""End-to-end RAG: question -> retrieved contexts -> grounded answer."""
from __future__ import annotations

import sys
from dataclasses import dataclass

from rag import config
from rag.generator import generate
from rag.vector_store import retrieve


@dataclass(frozen=True)
class RAGResult:
    question: str
    answer: str
    retrieval_context: list[str]
    sources: list[str]


def answer(question: str, top_k: int = config.TOP_K) -> RAGResult:
    chunks = retrieve(question, top_k=top_k)
    contexts = [c.text for c in chunks]
    return RAGResult(
        question=question,
        answer=generate(question, contexts),
        retrieval_context=contexts,
        sources=[c.source for c in chunks],
    )


if __name__ == '__main__':
    q = ' '.join(sys.argv[1:]) or 'How long do I have to get a full refund?'
    result = answer(q)
    print(f'Q: {result.question}\nA: {result.answer}\n\nSources: {result.sources}')
