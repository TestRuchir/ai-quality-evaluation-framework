"""Regression tracking: compare this run's per-metric scores against a committed baseline, and keep a history.

  python -m evals.regression check     # exit 1 if any metric regressed beyond tolerance (CI gate)
  python -m evals.regression update    # promote the latest run (reports/*_results.json) to the new baseline

The baseline (baselines/eval_baseline.json) is committed, so a pull request that makes the bot worse fails CI
even when every individual test is still above its absolute threshold.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from evals.report import REPORTS_DIR, load_rows, summarize
from rag import config

BASELINE_PATH = config.ROOT / 'baselines' / 'eval_baseline.json'
HISTORY_PATH = REPORTS_DIR / 'history.jsonl'

# Allowed drop before a change counts as a regression. LLM judges are noisy, so this is not zero.
SCORE_TOLERANCE = 0.05
PASS_RATE_TOLERANCE = 0.10
# DeepEval metrics where a LOWER score is better.
LOWER_IS_BETTER = {'Hallucination', 'Toxicity', 'Bias'}


def key(suite: str, metric: str) -> str:
    return f'{suite}/{metric}'


def summary_by_key(rows: list[dict]) -> dict[str, dict]:
    return {key(s['suite'], s['metric']): {'avg_score': s['avg_score'], 'pass_rate': s['pass_rate'], 'n': s['n']}
            for s in summarize(rows)}


def compare(current: dict[str, dict], baseline: dict[str, dict],
            score_tol: float = SCORE_TOLERANCE, pass_tol: float = PASS_RATE_TOLERANCE) -> list[dict]:
    """One entry per metric present in both runs, with deltas and whether it regressed."""
    results = []
    for k in sorted(current.keys() & baseline.keys()):
        cur, base = current[k], baseline[k]
        metric = k.split('/', 1)[1]
        sign = -1 if metric in LOWER_IS_BETTER else 1
        score_delta = cur['avg_score'] - base['avg_score']
        pass_delta = cur['pass_rate'] - base['pass_rate']
        regressed = sign * score_delta < -score_tol or pass_delta < -pass_tol
        results.append({'key': k, 'score_delta': score_delta, 'pass_rate_delta': pass_delta, 'regressed': regressed})
    return results


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, dict]:
    return json.loads(path.read_text())['metrics'] if path.exists() else {}


def write_baseline(current: dict[str, dict], path: Path = BASELINE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'updated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'generation_model': config.GENERATION_MODEL,
        'judge_model': config.JUDGE_MODEL,
        'metrics': current,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return path


def append_history(current: dict[str, dict], path: Path = HISTORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {'time': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'metrics': current}
    with path.open('a') as f:
        f.write(json.dumps(entry, sort_keys=True) + '\n')


def main(argv: list[str]) -> int:
    command = argv[0] if argv else 'check'
    current = summary_by_key(load_rows())
    if not current:
        print('No results in reports/; run python -m evals.run_evaluation first.')
        return 1
    if command == 'update':
        print(f'Baseline written to {write_baseline(current)} ({len(current)} metrics)')
        return 0
    if command != 'check':
        raise SystemExit('usage: python -m evals.regression [check|update]')

    baseline = load_baseline()
    if not baseline:
        print(f'No baseline at {BASELINE_PATH}; run "python -m evals.regression update" to create one.')
        return 0
    comparison = compare(current, baseline)
    print(f'{"Metric":<48}{"Score Δ":>9}{"Pass Δ":>9}  Status')
    for c in comparison:
        status = 'REGRESSED' if c['regressed'] else 'ok'
        print(f'{c["key"]:<48}{c["score_delta"]:>+9.2f}{c["pass_rate_delta"]:>+9.0%}  {status}')
    for k in sorted(baseline.keys() - current.keys()):
        print(f'{k:<48}{"":>18}  missing from this run')
    regressions = [c for c in comparison if c['regressed']]
    print(f'\n{len(regressions)} regression(s) against baseline from {json.loads(BASELINE_PATH.read_text())["updated"]}')
    return 1 if regressions else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
