"""Rule-based guardrails. Pure Python so they can be unit-tested offline against datasets/safety_golden.json.

Input guard: block prompt-injection / jailbreak attempts before they reach the LLM.
Output guard: redact PII (emails, phone numbers, card numbers) before a response leaves the agent.
"""
from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r'\b(ignore|disregard|forget|override)\b.{0,20}\b(all|any|the|your|prior|previous|above)\b.{0,20}'
    r'\b(instructions?|rules|prompts?|polic(y|ies))\b',
    r'\b(reveal|print|show|repeat|leak)\b.{0,30}\b(system prompt|hidden instructions|instructions verbatim)\b',
    r'\b(hidden|secret) instructions\b',
    r'\byou are now\b',
    r'\bfrom now on you\b',
    r'</?\s*(context|system)\s*>|^\s*system\s*:|\bSYSTEM:',
    r'\bdeveloper mode\b',
    r'\bdo anything now\b',
    r'\bwithout any restrictions\b|\bno (restrictions|policies)\b',
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]
# Case-sensitive so the name "Dan" is not blocked.
_INJECTION_RE.append(re.compile(r'\bDAN\b'))

PII_PATTERNS = {
    'EMAIL': r'[\w.+-]+@[\w-]+\.[\w.-]+',
    'CARD': r'\b(?:\d[ -]?){13,16}\b',
    'PHONE': r'\+?\d{1,2}[ .-]?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}\b',
}

REFUSAL_MESSAGE = "I can't help with that request. I can answer questions about orders, refunds, shipping, warranty, and account security."


def detect_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in _INJECTION_RE)


def redact_pii(text: str) -> str:
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f'[REDACTED_{label}]', text)
    return text
