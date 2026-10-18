"""Offline checks for HTML reporting and judge/human agreement math."""
import pytest

from evals.human_review import agreement, cohen_kappa, parse_label
from evals.report import render_html, summarize

ROWS = [
    {'suite': 'rag', 'case_id': 'refund-window', 'metric': 'Faithfulness', 'score': 1.0, 'threshold': 0.8, 'success': True, 'reason': 'ok'},
    {'suite': 'rag', 'case_id': 'gift-card', 'metric': 'Faithfulness', 'score': 0.5, 'threshold': 0.8, 'success': False, 'reason': '<b>bad</b>'},
    {'suite': 'safety', 'case_id': 'inj-1', 'metric': 'Toxicity', 'score': 0.0, 'threshold': 0.2, 'success': True, 'reason': None},
]


def test_summarize_groups_by_suite_and_metric():
    summary = {(s['suite'], s['metric']): s for s in summarize(ROWS)}
    assert summary[('rag', 'Faithfulness')]['n'] == 2
    assert summary[('rag', 'Faithfulness')]['avg_score'] == pytest.approx(0.75)
    assert summary[('rag', 'Faithfulness')]['pass_rate'] == 0.5


def test_render_html_lists_failures_and_escapes_reasons():
    html = render_html(ROWS)
    assert html.startswith('<!doctype html>')
    assert '67%' in html  # 2 of 3 checks pass
    assert '<b>bad</b>' not in html and '&lt;b&gt;bad&lt;/b&gt;' in html
    assert html.index('gift-card') < html.index('refund-window')  # failures first


def test_render_html_handles_empty_results():
    html = render_html([])
    assert '<b>0</b>metric checks' in html
    assert '<b>0%</b>pass rate' in html


def test_cohen_kappa_known_values():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
    # 20 items: 7 both-yes, 8 both-no, 5 disagreements -> kappa 0.5
    a = [True] * 7 + [False] * 8 + [True] * 3 + [False] * 2
    b = [True] * 7 + [False] * 8 + [False] * 3 + [True] * 2
    assert cohen_kappa(a, b) == pytest.approx(0.5, abs=0.01)


def test_agreement_skips_unreviewed_rows():
    review = [
        {'metric': 'Faithfulness', 'judge_pass': 'yes', 'human_pass': 'yes'},
        {'metric': 'Faithfulness', 'judge_pass': 'no', 'human_pass': 'yes'},
        {'metric': 'Faithfulness', 'judge_pass': 'yes', 'human_pass': ''},
    ]
    stats = agreement(review)
    assert stats['Faithfulness']['n'] == 2
    assert stats['Faithfulness']['agreement'] == 0.5


def test_parse_label():
    assert parse_label(' Yes ') is True
    assert parse_label('fail') is False
    assert parse_label('') is None
