"""Live LangGraph agent with a real LLM, scored three ways:
  - white-box: deterministic assertions on the captured trace (tool order, arguments, step count)
  - DeepEval ToolCorrectnessMetric: tools called vs expected tools (deterministic, no LLM call)
  - DeepEval TaskCompletionMetric + GEval Correctness: did the agent accomplish the task (LLM judge)

Run with:  python -m pytest evals/test_agent_live.py -v
"""
from functools import lru_cache

import pytest
from deepeval import assert_test
from deepeval.metrics import TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from agent.graph import AgentResult, run_agent
from evals.claude_judge import judge
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
    {'id': 'multi-step-return', 'input': 'Can I still get a full refund for order 1002?', 'tools': ['get_order_status', 'search_policies'],
     'args': None, 'expected_output': 'Yes. Order 1002 was delivered 12 days ago, which is within the 30-day full refund window.'},
]

AGENT_THRESHOLDS = {'tool_correctness': 1.0, 'task_completion': 0.7}


@lru_cache(maxsize=None)
def run(task_id: str) -> AgentResult:
    task = next(t for t in TASKS if t['id'] == task_id)
    return run_agent(task['input'], customer_id='C001')


def build_agent_test_case(task: dict, result: AgentResult) -> LLMTestCase:
    return LLMTestCase(
        input=task['input'],
        actual_output=result.answer,
        expected_output=task['expected_output'],
        tools_called=[ToolCall(name=s['tool'], input_parameters=s['args'], output=s['output']) for s in result.trace],
        expected_tools=[ToolCall(name=name) for name in task['tools']],
        name=task['id'],
    )


def agent_metrics():
    return [
        ToolCorrectnessMetric(threshold=AGENT_THRESHOLDS['tool_correctness'], model=judge()),
        TaskCompletionMetric(threshold=AGENT_THRESHOLDS['task_completion'], model=judge()),
        correctness_metric(),
    ]


@pytest.mark.parametrize('task', TASKS, ids=[t['id'] for t in TASKS])
def test_agent_trace(task):
    """White-box: inspect the intermediate steps, not just the final answer."""
    result = run(task['id'])
    tools_used = [s['tool'] for s in result.trace]
    assert set(task['tools']) <= set(tools_used), f'expected {task["tools"]}, got {tools_used}'
    assert len(tools_used) <= 3, f'too many tool calls: {tools_used}'
    assert not result.hit_step_limit
    if task['args']:
        assert result.trace[0]['args'] == task['args']


@pytest.mark.parametrize('task', TASKS, ids=[t['id'] for t in TASKS])
def test_agent_task_completion(task):
    assert_test(build_agent_test_case(task, run(task['id'])), agent_metrics())
