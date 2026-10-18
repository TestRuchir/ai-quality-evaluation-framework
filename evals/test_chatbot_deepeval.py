"""Multi-turn evaluation of the live Claude chatbot with DeepEval conversational metrics.

Each ConversationalGolden scripts the user's turns; the bot plays them out, and the full conversation is
scored for knowledge retention, turn relevancy, role adherence, and conversation completeness.
Run with:  deepeval test run evals/test_chatbot_deepeval.py
"""
from functools import lru_cache

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    ConversationCompletenessMetric,
    KnowledgeRetentionMetric,
    RoleAdherenceMetric,
    TurnRelevancyMetric,
)
from deepeval.test_case import ConversationalTestCase, Turn

from chatbot.bot import SupportChatbot
from evals.claude_judge import judge
from evals.datasets import golden_id, load_conversation_dataset

CHATBOT_ROLE = 'A concise customer support assistant for an online store that only answers store-policy questions.'
CONVERSATIONS = load_conversation_dataset().goldens

CONVERSATION_THRESHOLDS = {
    'knowledge_retention': 0.7,
    'turn_relevancy': 0.7,
    'role_adherence': 0.7,
    'conversation_completeness': 0.6,
}


def conversation_metrics():
    t = CONVERSATION_THRESHOLDS
    return [
        KnowledgeRetentionMetric(threshold=t['knowledge_retention'], model=judge()),
        TurnRelevancyMetric(threshold=t['turn_relevancy'], model=judge()),
        RoleAdherenceMetric(threshold=t['role_adherence'], model=judge()),
        ConversationCompletenessMetric(threshold=t['conversation_completeness'], model=judge()),
    ]


@lru_cache(maxsize=None)
def play(conversation_id: str) -> tuple[Turn, ...]:
    golden = next(g for g in CONVERSATIONS if golden_id(g) == conversation_id)
    bot = SupportChatbot()
    turns = []
    for message in golden.additional_metadata['user_turns']:
        turns += [Turn(role='user', content=message), Turn(role='assistant', content=bot.reply(conversation_id, message))]
    return tuple(turns)


def build_conversational_test_case(golden) -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=list(play(golden_id(golden))),
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
        chatbot_role=CHATBOT_ROLE,
        name=golden_id(golden),
    )


@pytest.mark.parametrize('golden', CONVERSATIONS, ids=[golden_id(g) for g in CONVERSATIONS])
def test_final_turn_uses_earlier_context(golden):
    """Deterministic check: facts from earlier turns show up in the bot's last reply."""
    final_reply = play(golden_id(golden))[-1].content
    for fact in golden.additional_metadata['must_mention']:
        assert fact.lower() in final_reply.lower(), f'expected {fact!r} in final reply: {final_reply!r}'


@pytest.mark.parametrize('golden', CONVERSATIONS, ids=[golden_id(g) for g in CONVERSATIONS])
def test_conversation_quality(golden):
    assert_test(build_conversational_test_case(golden), conversation_metrics())
