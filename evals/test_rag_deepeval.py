"""DeepEval evaluation of the live RAG pipeline against the golden dataset.

Retrieval and generation are scored in separate tests so a failure points at the component that broke.
Run with:  deepeval test run evals/   (or: python -m pytest evals -v)
"""
import json
from functools import lru_cache

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from evals.metrics import generation_metrics, refusal_metric, retrieval_metrics
from rag import config
from rag.pipeline import RAGResult, answer
from rag.vector_store import retrieve

GOLDENS = json.loads((config.ROOT / 'datasets' / 'rag_golden.json').read_text())
IN_SCOPE = [g for g in GOLDENS if g['category'] == 'in_scope']
OUT_OF_SCOPE = [g for g in GOLDENS if g['category'] == 'out_of_scope']


@lru_cache(maxsize=None)
def run_pipeline(question: str) -> RAGResult:
    """Cache so retrieval and generation tests share one pipeline call per question."""
    return answer(question)


def build_test_case(golden: dict) -> LLMTestCase:
    result = run_pipeline(golden['input'])
    return LLMTestCase(
        input=golden['input'],
        actual_output=result.answer,
        expected_output=golden['expected_output'],
        retrieval_context=result.retrieval_context,  # what the retriever returned (Faithfulness, Contextual*)
        context=golden['context'] or None,  # ground-truth facts (Hallucination)
    )


@pytest.mark.parametrize('golden', IN_SCOPE, ids=[g['id'] for g in IN_SCOPE])
def test_retrieval_hit_rate(golden):
    """Cheap deterministic check (embeddings only, no LLM judge): expected doc appears in top-k."""
    sources = {c.source for c in retrieve(golden['input'])}
    assert set(golden['expected_sources']) <= sources, f'expected {golden["expected_sources"]}, got {sources}'


@pytest.mark.parametrize('golden', IN_SCOPE, ids=[g['id'] for g in IN_SCOPE])
def test_retrieval_quality(golden):
    assert_test(build_test_case(golden), retrieval_metrics())


@pytest.mark.parametrize('golden', IN_SCOPE, ids=[g['id'] for g in IN_SCOPE])
def test_generation_quality(golden):
    assert_test(build_test_case(golden), generation_metrics())


@pytest.mark.parametrize('golden', OUT_OF_SCOPE, ids=[g['id'] for g in OUT_OF_SCOPE])
def test_out_of_scope_refusal(golden):
    assert_test(build_test_case(golden), [refusal_metric()])
