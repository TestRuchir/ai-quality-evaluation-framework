"""Black-box exact-match benchmark on synthetic, deterministic answers."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'datasets' / 'golden_dataset.json'


def exact_match(actual: str, expected: str) -> bool:
    return actual.strip().casefold() == expected.strip().casefold()


def aggregate_accuracy(rows: list[dict]) -> float:
    return sum(exact_match(r['actual'], r['expected']) for r in rows) / len(rows) if rows else 0.0


def test_golden_answers():
    rows = json.loads(DATA.read_text())
    assert all(exact_match(r['actual'], r['expected']) for r in rows)
    assert aggregate_accuracy(rows) == 1.0


def test_incorrect_answer_is_detected():
    assert not exact_match('73', '72')
