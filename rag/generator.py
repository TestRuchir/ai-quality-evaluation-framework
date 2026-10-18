"""Grounded answer generation with Claude."""
from __future__ import annotations

import anthropic

from rag import config

SYSTEM_PROMPT = (
    'You are a customer support assistant. Answer the question using ONLY the provided context. '
    'If the context does not contain the answer, say "I don\'t have that information." '
    'Do not use outside knowledge. Be concise.'
)
REFUSAL_ANSWER = "I can't help with that request."

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return _client


def generate(question: str, contexts: list[str]) -> str:
    context_block = '\n\n'.join(f'[{i}] {c}' for i, c in enumerate(contexts, start=1))
    response = _get_client().beta.messages.create(
        model=config.GENERATION_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': f'Context:\n{context_block}\n\nQuestion: {question}'}],
        # If a safety classifier declines, Anthropic re-runs the request on its recommended fallback model.
        betas=['server-side-fallback-2026-07-01'],
        fallbacks='default',
    )
    if response.stop_reason == 'refusal':
        return REFUSAL_ANSWER
    return ''.join(b.text for b in response.content if b.type == 'text').strip()
