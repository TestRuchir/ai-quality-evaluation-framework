"""RAGAS metrics alongside DeepEval: a second, independently-implemented judge for the same RAG pipeline.

Agreement between the two frameworks is evidence the score reflects the system, not one judge's quirks.
"""
from __future__ import annotations

import asyncio

from anthropic import AsyncAnthropic
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from rag import config

THRESHOLDS = {
    'faithfulness': 0.8,
    'answer_relevancy': 0.7,
    'context_precision': 0.7,
    'context_recall': 0.7,
}


class LocalEmbeddings(BaseRagasEmbedding):
    """Same local embedding model the vector store uses, so AnswerRelevancy needs no embeddings API."""

    def __init__(self):
        super().__init__()
        self._fn = DefaultEmbeddingFunction()

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return [float(x) for x in self._fn([text])[0]]

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return await asyncio.to_thread(self.embed_text, text)


def build_metrics():
    llm = llm_factory(config.JUDGE_MODEL, provider='anthropic', client=AsyncAnthropic())
    # RAGAS defaults to temperature/top_p, which current Claude models reject; give the judge room to think.
    llm.model_args.pop('temperature', None)
    llm.model_args.pop('top_p', None)
    llm.model_args['max_tokens'] = 16000
    embeddings = LocalEmbeddings()
    return {
        'faithfulness': Faithfulness(llm=llm),
        'answer_relevancy': AnswerRelevancy(llm=llm, embeddings=embeddings),
        'context_precision': ContextPrecision(llm=llm),
        'context_recall': ContextRecall(llm=llm),
    }


async def ascore(metrics: dict, question: str, answer: str, contexts: list[str], reference: str) -> dict[str, float]:
    results = await asyncio.gather(
        metrics['faithfulness'].ascore(user_input=question, response=answer, retrieved_contexts=contexts),
        metrics['answer_relevancy'].ascore(user_input=question, response=answer),
        metrics['context_precision'].ascore(user_input=question, reference=reference, retrieved_contexts=contexts),
        metrics['context_recall'].ascore(user_input=question, retrieved_contexts=contexts, reference=reference),
    )
    return {name: float(r.value) for name, r in zip(metrics, results)}


def score(metrics: dict, question: str, answer: str, contexts: list[str], reference: str) -> dict[str, float]:
    return asyncio.run(ascore(metrics, question, answer, contexts, reference))
