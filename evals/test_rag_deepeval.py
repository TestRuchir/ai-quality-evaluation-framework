"""DeepEval evaluation of the live RAG pipeline against the golden dataset.

Retrieval and generation are scored in separate tests so a failure points at the component that broke.
Run with:  deepeval test run evals/   (or: python -m pytest evals -v)
"""
from functools import lru_cache

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval.dataset import Golden

from evals.datasets import golden_id, load_rag_dataset
from evals.metrics import generation_metrics, refusal_metric, retrieval_metrics
from rag.pipeline import RAGResult, answer
from rag.vector_store import retrieve

IN_SCOPE = load_rag_dataset('in_scope').goldens
OUT_OF_SCOPE = load_rag_dataset('out_of_scope').goldens


@lru_cache(maxsize=None)
def run_pipeline(question: str) -> RAGResult:
    """Cache so retrieval and generation tests share one pipeline call per question."""
    return answer(question)


def build_test_case(golden: Golden) -> LLMTestCase:
    result = run_pipeline(golden.input)
    return LLMTestCase(
        input=golden.input,
        actual_output=result.answer,
        expected_output=golden.expected_output,
        retrieval_context=result.retrieval_context,  # what the retriever returned (Faithfulness, Contextual*)
        context=golden.context,  # ground-truth facts (Hallucination)
        name=golden_id(golden),
    )


@pytest.mark.parametrize('golden', IN_SCOPE, ids=[golden_id(g) for g in IN_SCOPE])
def test_retrieval_hit_rate(golden):
    """Cheap deterministic check (embeddings only, no LLM judge): expected doc appears in top-k."""
    expected = set(golden.additional_metadata['expected_sources'])
    sources = {c.source for c in retrieve(golden.input)}
    assert expected <= sources, f'expected {expected}, got {sources}'


@pytest.mark.parametrize('golden', IN_SCOPE, ids=[golden_id(g) for g in IN_SCOPE])
def test_retrieval_quality(golden):
    assert_test(build_test_case(golden), retrieval_metrics())


@pytest.mark.parametrize('golden', IN_SCOPE, ids=[golden_id(g) for g in IN_SCOPE])
def test_generation_quality(golden):
    assert_test(build_test_case(golden), generation_metrics())


@pytest.mark.parametrize('golden', OUT_OF_SCOPE, ids=[golden_id(g) for g in OUT_OF_SCOPE])
def test_out_of_scope_refusal(golden):
    assert_test(build_test_case(golden), [refusal_metric()])
