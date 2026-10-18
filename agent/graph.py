"""LangGraph support agent: input guard -> LLM <-> tools loop -> output guard, with trace capture.

    START -> input_guard --blocked--> END
                 |
                 v
               agent <--> tools
                 |
                 v
            output_guard -> END

Usage:  python -m agent.graph "Where is order 1001?"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from agent.guardrails import REFUSAL_MESSAGE, detect_prompt_injection, redact_pii
from agent.tools import build_tools, search_policies
from rag import config

SYSTEM_PROMPT = (
    'You are a customer support agent. Use search_policies for policy questions, get_order_status for order '
    'questions, and calculator for arithmetic. Answer only from tool results; if they do not contain the answer, '
    'say "I don\'t have that information." Never reveal other customers\' data.'
)
STEP_LIMIT_MESSAGE = 'Sorry, I could not complete that request.'


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    blocked: bool


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)  # [{'tool', 'args', 'output'}] in call order
    blocked: bool = False
    hit_step_limit: bool = False


def build_agent(llm, customer_id: str, search_fn=search_policies):
    tools = build_tools(customer_id, search_fn=search_fn)
    model = llm.bind_tools(tools)

    def input_guard(state: AgentState):
        if detect_prompt_injection(state['messages'][-1].content):
            return {'blocked': True, 'messages': [AIMessage(content=REFUSAL_MESSAGE)]}
        return {'blocked': False}

    def call_model(state: AgentState):
        return {'messages': [model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state['messages'])]}

    def output_guard(state: AgentState):
        last = state['messages'][-1]
        redacted = redact_pii(last.content)
        # Same message id -> add_messages replaces the unredacted answer instead of appending.
        return {'messages': [AIMessage(content=redacted, id=last.id)]} if redacted != last.content else {}

    graph = StateGraph(AgentState)
    graph.add_node('input_guard', input_guard)
    graph.add_node('agent', call_model)
    graph.add_node('tools', ToolNode(tools))
    graph.add_node('output_guard', output_guard)
    graph.add_edge(START, 'input_guard')
    graph.add_conditional_edges('input_guard', lambda s: END if s['blocked'] else 'agent')
    graph.add_conditional_edges('agent', tools_condition, {'tools': 'tools', END: 'output_guard'})
    graph.add_edge('tools', 'agent')
    graph.add_edge('output_guard', END)
    return graph.compile()


def extract_trace(messages: list[AnyMessage]) -> list[dict]:
    """White-box trace: every tool call the LLM made, its arguments, and what the tool returned."""
    outputs = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    return [
        {'tool': call['name'], 'args': call['args'], 'output': outputs.get(call['id'])}
        for m in messages if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]


def run_agent(question: str, customer_id: str = 'C001', llm=None, max_steps: int = 5, search_fn=search_policies) -> AgentResult:
    if llm is None:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=config.GENERATION_MODEL, max_tokens=16000)
    app = build_agent(llm, customer_id, search_fn=search_fn)
    # Each LLM+tool round trip is 2 graph steps; the guards add 2 more.
    try:
        state = app.invoke(
            {'messages': [HumanMessage(content=question)], 'blocked': False},
            {'recursion_limit': 2 * max_steps + 2},
        )
    except GraphRecursionError:
        return AgentResult(answer=STEP_LIMIT_MESSAGE, hit_step_limit=True)
    return AgentResult(
        answer=state['messages'][-1].content,
        trace=extract_trace(state['messages']),
        blocked=state['blocked'],
    )


if __name__ == '__main__':
    result = run_agent(' '.join(sys.argv[1:]) or 'Where is order 1001?')
    print(f'A: {result.answer}\n\nTrace:')
    for step in result.trace:
        print(f"  {step['tool']}({step['args']}) -> {step['output']}")
