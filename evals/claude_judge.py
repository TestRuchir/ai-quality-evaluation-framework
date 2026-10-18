"""Claude as the DeepEval judge, via the official Anthropic SDK.

DeepEval's built-in AnthropicModel always sends `temperature`, which current Claude models reject,
so this thin wrapper is used instead. Structured outputs guarantee the judge returns valid JSON.
"""
from __future__ import annotations

import anthropic
from deepeval.models import DeepEvalBaseLLM

from rag.config import JUDGE_MODEL


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = JUDGE_MODEL):
        self.model_name = model
        self.client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.model_name

    def _request(self, prompt: str) -> dict:
        return {'model': self.model_name, 'max_tokens': 16000, 'messages': [{'role': 'user', 'content': prompt}]}

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self.client.messages.parse(**self._request(prompt), output_format=schema).parsed_output
        response = self.client.messages.create(**self._request(prompt))
        return ''.join(b.text for b in response.content if b.type == 'text')

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            response = await self.async_client.messages.parse(**self._request(prompt), output_format=schema)
            return response.parsed_output
        response = await self.async_client.messages.create(**self._request(prompt))
        return ''.join(b.text for b in response.content if b.type == 'text')


_judge: ClaudeJudge | None = None


def judge() -> ClaudeJudge:
    """Shared judge instance so all metrics reuse one HTTP client."""
    global _judge
    if _judge is None:
        _judge = ClaudeJudge()
    return _judge
