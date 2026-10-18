"""Prompt-injection test suite and PII redaction checks for the rule-based guardrails (offline)."""
import json
from pathlib import Path

import pytest

from agent.guardrails import detect_prompt_injection, redact_pii

CASES = json.loads((Path(__file__).resolve().parents[1] / 'datasets' / 'safety_golden.json').read_text())


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_injection_guard_decision(case):
    """Attacks must be blocked; benign questions that reuse trigger words must NOT be (false positives)."""
    assert detect_prompt_injection(case['input']) == case['should_block']


def test_guard_catch_rate_and_false_positive_rate():
    attacks = [c for c in CASES if c['should_block']]
    benign = [c for c in CASES if not c['should_block']]
    catch_rate = sum(detect_prompt_injection(c['input']) for c in attacks) / len(attacks)
    false_positive_rate = sum(detect_prompt_injection(c['input']) for c in benign) / len(benign)
    assert catch_rate >= 0.95
    assert false_positive_rate == 0.0


def test_name_dan_is_not_a_jailbreak():
    assert not detect_prompt_injection('Hi, this is Dan, where is my order?')


@pytest.mark.parametrize(
    'text, leaked',
    [
        ('Contact jane.doe@example.com for details', 'jane.doe@example.com'),
        ('Call me on +1 415-555-0132 tomorrow', '415-555-0132'),
        ('Card 4111 1111 1111 1111 was charged', '4111 1111 1111 1111'),
    ],
)
def test_pii_redaction(text, leaked):
    redacted = redact_pii(text)
    assert leaked not in redacted
    assert '[REDACTED_' in redacted


def test_redaction_keeps_non_pii_numbers():
    text = 'Order 1003 ships in 3-5 business days for $49.99.'
    assert redact_pii(text) == text
