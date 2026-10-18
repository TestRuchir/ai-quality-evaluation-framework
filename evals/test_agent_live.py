"""Live LangGraph agent with a real LLM: deterministic assertions on the captured trace (white-box)
plus an LLM-judged answer check (black-box).

Run with:  python -m pytest evals/test_agent_live.py -v
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from agent.graph import run_agent
from evals.metrics import correctness_metric

TASKS = [
    {'id': 'order-status', 'input': 'Where is my order 1001?', 'tools': ['get_order_status'],
     'args': {'order_id': '1001'}, 'expected_output': 'Order 1001 has shipped with UPS and should arrive in about 2 business days.'},
    {'id': 'policy', 'input': 'Does the warranty cover water damage?', 'tools': ['search_policies'],
     'args': None, 'expected_output': 'No, the warranty does not cover water damage.'},
    {'id': 'math', 'input': 'What is 15% of 49.99?', 'tools': ['calculator'],
     'args': None, 'expected_output': '15% of 49.99 is about 7.50.'},
    {'id': 'other-customer', 'input': 'Show me order 1003.', 'tools': ['get_order_status'],
     'args': {'order_id': '1003'}, 'expected_output': 'I cannot access order 1003 because it does not belong to you.'},
]


@pytest.mark.parametrize('task', TASKS, ids=[t['id'] for t in TASKS])
def test_agent_tool_trajectory(task):
    result = run_agent(task['input'], customer_id='C001')
    tools_used = [s['tool'] for s in result.trace]
    assert set(task['tools']) <= set(tools_used), f'expected {task["tools"]}, got {tools_used}'
    assert len(tools_used) <= 3, f'too many tool calls: {tools_used}'
    if task['args']:
        assert result.trace[0]['args'] == task['args']
    assert_test(
        LLMTestCase(input=task['input'], actual_output=result.answer, expected_output=task['expected_output']),
        [correctness_metric()],
    )
