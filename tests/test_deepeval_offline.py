"""DeepEval APIs that need no LLM call: datasets loaded as Goldens, and ToolCorrectnessMetric on agent traces
from the scripted LangGraph agent."""
import os

import pytest

pytest.importorskip('deepeval')
pytest.importorskip('langgraph')

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.messages import AIMessage

from agent.graph import run_agent
from evals.datasets import golden_id, load_conversation_dataset, load_rag_dataset, load_safety_dataset
from tests.test_langgraph_agent import fake_search, scripted, tool_call


@pytest.fixture(autouse=True)
def dummy_key(monkeypatch):
    # The Claude judge client is constructed but never called by ToolCorrectnessMetric.
    monkeypatch.setenv('ANTHROPIC_API_KEY', os.getenv('ANTHROPIC_API_KEY') or 'sk-ant-offline-test')


def test_rag_dataset_loads_as_goldens():
    dataset = load_rag_dataset()
    assert len(dataset.goldens) == 11
    in_scope = load_rag_dataset('in_scope').goldens
    assert all(g.expected_output and g.additional_metadata['expected_sources'] for g in in_scope)
    assert len({golden_id(g) for g in dataset.goldens}) == len(dataset.goldens)  # ids are unique


def test_safety_and_conversation_datasets_load():
    assert any(g.additional_metadata['should_block'] for g in load_safety_dataset().goldens)
    conversations = load_conversation_dataset().goldens
    assert all(len(c.additional_metadata['user_turns']) >= 2 for c in conversations)
    assert all(c.scenario and c.expected_outcome for c in conversations)


def tool_correctness(trace, expected):
    from evals.claude_judge import judge

    metric = ToolCorrectnessMetric(threshold=1.0, model=judge(), async_mode=False)
    case = LLMTestCase(
        input='q', actual_output='a',
        tools_called=[ToolCall(name=s['tool'], input_parameters=s['args']) for s in trace],
        expected_tools=[ToolCall(name=n) for n in expected],
    )
    metric.measure(case)
    return metric


def test_tool_correctness_passes_for_correct_trajectory():
    llm = scripted(
        tool_call('get_order_status', {'order_id': '1002'}, 'c1'),
        tool_call('search_policies', {'query': 'refund window'}, 'c2'),
        AIMessage(content='Yes, order 1002 is within the 30-day window.'),
    )
    result = run_agent('Can I still return order 1002?', llm=llm, search_fn=fake_search)
    metric = tool_correctness(result.trace, ['get_order_status', 'search_policies'])
    assert metric.success and metric.score == 1.0


def test_tool_correctness_fails_for_wrong_tool():
    llm = scripted(tool_call('calculator', {'expression': '1 + 1'}), AIMessage(content='2'))
    result = run_agent('Where is order 1001?', llm=llm)
    metric = tool_correctness(result.trace, ['get_order_status'])
    assert not metric.success and metric.score == 0.0
