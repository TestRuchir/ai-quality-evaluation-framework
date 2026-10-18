"""RAGAS evaluation of the live RAG pipeline (same goldens and pipeline cache as the DeepEval tests).

Run with:  python -m pytest evals/test_rag_ragas.py -v
"""
from functools import lru_cache

import pytest

from evals.ragas_metrics import THRESHOLDS, build_metrics, score
from evals.test_rag_deepeval import IN_SCOPE, run_pipeline


@lru_cache(maxsize=1)
def metrics():
    return build_metrics()


def ragas_scores(golden: dict) -> dict[str, float]:
    result = run_pipeline(golden['input'])
    return score(metrics(), golden['input'], result.answer, result.retrieval_context, golden['expected_output'])


@pytest.mark.parametrize('golden', IN_SCOPE, ids=[g['id'] for g in IN_SCOPE])
def test_ragas_quality(golden):
    scores = ragas_scores(golden)
    failures = {m: round(s, 2) for m, s in scores.items() if s < THRESHOLDS[m]}
    assert not failures, f'below threshold {THRESHOLDS}: {failures}'
