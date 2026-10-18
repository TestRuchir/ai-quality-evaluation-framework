"""Offline checks for regression tracking: deltas, metric direction, tolerances, baseline and history files."""
import json

import pytest

from evals.regression import append_history, compare, load_baseline, summary_by_key, write_baseline
from evals.report import render_html

BASELINE = {
    'rag-deepeval/Faithfulness': {'avg_score': 0.90, 'pass_rate': 1.0, 'n': 9},
    'rag-deepeval/Hallucination': {'avg_score': 0.10, 'pass_rate': 1.0, 'n': 9},
    'agent/Tool Correctness': {'avg_score': 1.0, 'pass_rate': 1.0, 'n': 5},
}


def by_key(results):
    return {r['key']: r for r in results}


def test_no_change_is_not_a_regression():
    assert not any(r['regressed'] for r in compare(BASELINE, BASELINE))


def test_score_drop_beyond_tolerance_regresses():
    current = {**BASELINE, 'rag-deepeval/Faithfulness': {'avg_score': 0.80, 'pass_rate': 1.0, 'n': 9}}
    result = by_key(compare(current, BASELINE))['rag-deepeval/Faithfulness']
    assert result['regressed']
    assert result['score_delta'] == pytest.approx(-0.10)


def test_small_judge_noise_is_tolerated():
    current = {**BASELINE, 'rag-deepeval/Faithfulness': {'avg_score': 0.87, 'pass_rate': 1.0, 'n': 9}}
    assert not by_key(compare(current, BASELINE))['rag-deepeval/Faithfulness']['regressed']


def test_lower_is_better_metrics_regress_when_score_rises():
    worse = {**BASELINE, 'rag-deepeval/Hallucination': {'avg_score': 0.40, 'pass_rate': 1.0, 'n': 9}}
    better = {**BASELINE, 'rag-deepeval/Hallucination': {'avg_score': 0.0, 'pass_rate': 1.0, 'n': 9}}
    assert by_key(compare(worse, BASELINE))['rag-deepeval/Hallucination']['regressed']
    assert not by_key(compare(better, BASELINE))['rag-deepeval/Hallucination']['regressed']


def test_pass_rate_drop_regresses_even_if_average_holds():
    current = {**BASELINE, 'agent/Tool Correctness': {'avg_score': 0.98, 'pass_rate': 0.8, 'n': 5}}
    assert by_key(compare(current, BASELINE))['agent/Tool Correctness']['regressed']


def test_new_metrics_are_not_compared():
    current = {**BASELINE, 'chatbot/Knowledge Retention': {'avg_score': 0.5, 'pass_rate': 0.5, 'n': 4}}
    assert 'chatbot/Knowledge Retention' not in by_key(compare(current, BASELINE))


def test_summary_by_key_from_rows():
    rows = [
        {'suite': 'agent', 'case_id': 'a', 'metric': 'Tool Correctness', 'score': 1.0, 'threshold': 1.0, 'success': True},
        {'suite': 'agent', 'case_id': 'b', 'metric': 'Tool Correctness', 'score': 0.0, 'threshold': 1.0, 'success': False},
    ]
    assert summary_by_key(rows) == {'agent/Tool Correctness': {'avg_score': 0.5, 'pass_rate': 0.5, 'n': 2}}


def test_baseline_round_trip_and_history(tmp_path):
    path = write_baseline(BASELINE, tmp_path / 'baseline.json')
    assert load_baseline(path) == BASELINE
    assert load_baseline(tmp_path / 'missing.json') == {}
    history = tmp_path / 'history.jsonl'
    append_history(BASELINE, history)
    append_history(BASELINE, history)
    lines = history.read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])['metrics'] == BASELINE


def test_report_shows_regressions_vs_baseline():
    rows = [{'suite': 'rag-deepeval', 'case_id': 'x', 'metric': 'Faithfulness', 'score': 0.5, 'threshold': 0.8, 'success': False}]
    deltas = by_key(compare(summary_by_key(rows), BASELINE))
    html = render_html(rows, deltas=deltas)
    assert 'vs baseline' in html
    assert '<b>1</b>regressions vs baseline' in html
    assert '-0.40' in html
