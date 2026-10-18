"""Human review of LLM-judge verdicts, and judge/human agreement (Cohen's kappa).

Workflow:
  1. python -m evals.run_evaluation          -> reports/*_results.json
  2. python -m evals.human_review export     -> reports/human_review.csv (fill in the human_pass column: yes/no)
  3. python -m evals.human_review agreement  -> per-metric agreement and kappa

Low kappa on a metric means the judge cannot be trusted as a CI gate for it until its rubric is fixed.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from evals.report import REPORTS_DIR, load_rows

REVIEW_CSV = REPORTS_DIR / 'human_review.csv'
FIELDS = ['suite', 'case_id', 'metric', 'judge_score', 'judge_pass', 'judge_reason', 'human_pass', 'notes']
_TRUE = {'yes', 'y', 'true', '1', 'pass'}
_FALSE = {'no', 'n', 'false', '0', 'fail'}


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement between two raters, corrected for agreement expected by chance. 1 = perfect, 0 = chance."""
    if len(a) != len(b) or not a:
        raise ValueError('need two equal-length, non-empty label lists')
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:  # both raters gave one constant label
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def parse_label(value: str) -> bool | None:
    v = value.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return None  # not reviewed yet


def export(rows: list[dict], path: Path = REVIEW_CSV) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                'suite': r['suite'], 'case_id': r['case_id'], 'metric': r['metric'],
                'judge_score': f'{(r["score"] or 0.0):.2f}', 'judge_pass': 'yes' if r['success'] else 'no',
                'judge_reason': r.get('reason') or '', 'human_pass': '', 'notes': '',
            })
    return path


def agreement(review_rows: list[dict]) -> dict[str, dict]:
    """Per metric (plus 'ALL'): number of reviewed rows, raw agreement, and Cohen's kappa."""
    pairs = defaultdict(list)
    for r in review_rows:
        human = parse_label(r['human_pass'])
        if human is None:
            continue
        judge = parse_label(r['judge_pass'])
        pairs[r['metric']].append((judge, human))
        pairs['ALL'].append((judge, human))
    return {
        metric: {
            'n': len(ps),
            'agreement': sum(j == h for j, h in ps) / len(ps),
            'kappa': cohen_kappa([j for j, _ in ps], [h for _, h in ps]),
        }
        for metric, ps in pairs.items()
    }


def main(argv: list[str]) -> None:
    command = argv[0] if argv else 'export'
    if command == 'export':
        print(f'Wrote {export(load_rows())}; fill in the human_pass column, then run: python -m evals.human_review agreement')
    elif command == 'agreement':
        with REVIEW_CSV.open() as f:
            stats = agreement(list(csv.DictReader(f)))
        if not stats:
            print(f'No reviewed rows in {REVIEW_CSV} (human_pass is empty).')
            return
        print(f'{"Metric":<32}{"Reviewed":>9}{"Agree":>8}{"Kappa":>8}')
        for metric, s in sorted(stats.items()):
            print(f'{metric:<32}{s["n"]:>9}{s["agreement"]:>8.0%}{s["kappa"]:>8.2f}')
    else:
        raise SystemExit('usage: python -m evals.human_review [export|agreement]')


if __name__ == '__main__':
    main(sys.argv[1:])
