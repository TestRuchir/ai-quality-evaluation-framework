"""Multi-turn support chatbot: each session keeps its own message history; every turn is grounded in
freshly retrieved policy chunks. The client and retriever are injectable so tests run offline.

Usage:  python -m chatbot.bot   (interactive)
"""
from __future__ import annotations

from rag import config
from rag.generator import REFUSAL_ANSWER

SYSTEM_PROMPT = (
    'You are a customer support assistant for an online store. Use the conversation so far and the policy '
    'context provided with each message. If the context does not contain the answer, say "I don\'t have that '
    'information." Stay in your support role and decline unrelated requests. Be concise.'
)


def _default_retrieve(question: str) -> list[str]:
    from rag.vector_store import retrieve

    return [c.text for c in retrieve(question)]


class SupportChatbot:
    def __init__(self, client=None, retrieve_fn=_default_retrieve, model: str = config.GENERATION_MODEL):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.retrieve_fn = retrieve_fn
        self.model = model
        self.sessions: dict[str, list[dict]] = {}

    def history(self, session_id: str) -> list[dict]:
        return self.sessions.setdefault(session_id, [])

    def reply(self, session_id: str, message: str) -> str:
        history = self.history(session_id)
        context = '\n\n'.join(self.retrieve_fn(message))
        # History keeps the plain user message; retrieved context is attached only to the current turn.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=history + [{'role': 'user', 'content': f'Policy context:\n{context}\n\nCustomer: {message}'}],
        )
        if response.stop_reason == 'refusal':
            answer = REFUSAL_ANSWER
        else:
            answer = ''.join(b.text for b in response.content if b.type == 'text').strip()
        history += [{'role': 'user', 'content': message}, {'role': 'assistant', 'content': answer}]
        return answer


if __name__ == '__main__':
    bot = SupportChatbot()
    print('Support chatbot (Ctrl+C to exit)')
    while True:
        print(f'Bot: {bot.reply("cli", input("You: "))}\n')
