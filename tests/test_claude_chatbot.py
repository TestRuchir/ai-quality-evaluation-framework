"""Offline tests of the real Claude chatbot's memory handling, using a fake Anthropic client.

The fake client records every request, so the tests assert exactly what conversation history Claude would see.
"""
from types import SimpleNamespace

from chatbot.bot import SupportChatbot


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(stop_reason='end_turn', content=[SimpleNamespace(type='text', text=next(self.replies))])


def bot_with(*replies):
    client = FakeClient(replies)
    return SupportChatbot(client=client, retrieve_fn=lambda q: [f'context for: {q}'], model='test-model'), client


def test_history_is_sent_on_every_turn():
    bot, client = bot_with('Hi! How can I help with order 1002?', 'Your order number is 1002.')
    bot.reply('s1', 'I have a question about order 1002.')
    bot.reply('s1', 'Which order did I mention?')
    second = client.requests[1]['messages']
    assert [m['role'] for m in second] == ['user', 'assistant', 'user']
    assert second[0]['content'] == 'I have a question about order 1002.'
    assert second[1]['content'] == 'Hi! How can I help with order 1002?'


def test_sessions_are_isolated():
    bot, client = bot_with('Nice to meet you, Alex.', "I don't know your name yet.")
    bot.reply('a', 'My name is Alex.')
    bot.reply('b', 'What is my name?')
    assert len(client.requests[1]['messages']) == 1  # session b sees nothing from session a
    assert 'Alex' not in str(client.requests[1]['messages'])


def test_retrieved_context_only_on_current_turn():
    bot, client = bot_with('first', 'second')
    bot.reply('s', 'refund window?')
    bot.reply('s', 'and for gift cards?')
    messages = client.requests[1]['messages']
    assert 'context for: and for gift cards?' in messages[-1]['content']
    assert all('Policy context' not in m['content'] for m in messages[:-1])  # history stays clean


def test_refusal_is_replaced_with_safe_message():
    client = FakeClient([])
    client.create = lambda **kw: SimpleNamespace(stop_reason='refusal', content=[])
    bot = SupportChatbot(client=client, retrieve_fn=lambda q: [], model='test-model')
    assert bot.reply('s', 'something declined') == "I can't help with that request."
