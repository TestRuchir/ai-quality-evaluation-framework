"""Model-based safety evaluation of the live RAG pipeline and LangGraph agent on adversarial inputs.

The RAG pipeline has no input guard, so this measures how the raw model resists injection.
The agent has the rule-based guard, so this measures the combined defense.
Run with:  deepeval test run evals/test_safety_deepeval.py
"""
import json
from functools import lru_cache

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from agent.graph import run_agent
from evals.safety_metrics import injection_resistance_metric, safety_metrics
from evals.test_rag_deepeval import run_pipeline
from rag import config

CASES = json.loads((config.ROOT / 'datasets' / 'safety_golden.json').read_text())
ATTACKS = [c for c in CASES if c['category'] in ('prompt_injection', 'jailbreak')]


@lru_cache(maxsize=None)
def agent_answer(question: str) -> str:
    return run_agent(question, customer_id='C001').answer


@pytest.mark.parametrize('case', ATTACKS, ids=[c['id'] for c in ATTACKS])
def test_rag_resists_injection(case):
    output = run_pipeline(case['input']).answer
    assert_test(LLMTestCase(input=case['input'], actual_output=output), [injection_resistance_metric()])


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_agent_output_is_safe(case):
    """Toxicity, bias, and PII leakage on every case, including the PII-extraction and toxic-bait prompts."""
    assert_test(LLMTestCase(input=case['input'], actual_output=agent_answer(case['input'])), safety_metrics())


@pytest.mark.parametrize('case', ATTACKS, ids=[c['id'] for c in ATTACKS])
def test_agent_resists_injection(case):
    output = agent_answer(case['input'])
    assert_test(LLMTestCase(input=case['input'], actual_output=output), [injection_resistance_metric()])
