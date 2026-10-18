"""Small retrieval evaluation: not a live RAGAS/LLM integration."""

def precision(retrieved: set[str], relevant: set[str]) -> float:
    return len(retrieved & relevant) / len(retrieved) if retrieved else 0.0


def recall(retrieved: set[str], relevant: set[str]) -> float:
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def test_context_precision_recall():
    relevant = {'refund-policy', 'eligibility'}
    retrieved = {'refund-policy', 'eligibility', 'unrelated'}
    assert precision(retrieved, relevant) == 2 / 3
    assert recall(retrieved, relevant) == 1.0


def test_grounded_answer_uses_evidence():
    evidence = {'refund period': '30 days'}
    answer = 'The refund period is 30 days.'
    assert evidence['refund period'] in answer
    assert evidence['refund period'] == '30 days'
