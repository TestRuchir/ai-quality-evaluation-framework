"""Static HTML report from eval result rows. Pure Python (no LLM calls) so it is unit-tested offline.

A row is: {'suite', 'case_id', 'metric', 'score', 'threshold', 'success', 'reason'}.
Usage:  python -m evals.report   (merges reports/*_results.json into reports/eval_report.html)
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from rag import config

REPORTS_DIR = config.ROOT / 'reports'

CSS = """
:root { --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --pass:#1a7f37; --fail:#cf222e; --pass-bg:#e6f4ea; --fail-bg:#fde8e8; }
@media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#f2f2f7; --muted:#a1a1a6; --line:#2c2c2e; --pass:#4ac26b; --fail:#ff6b6b; --pass-bg:#12301c; --fail-bg:#3a1717; } }
body { background:var(--bg); color:var(--fg); font:14px/1.5 -apple-system, system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:24px 16px; }
h1 { margin:0 0 4px; } .muted { color:var(--muted); }
.tiles { display:flex; flex-wrap:wrap; gap:12px; margin:20px 0; }
.tile { border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:120px; }
.tile b { display:block; font-size:24px; }
.wrap { overflow-x:auto; } table { border-collapse:collapse; width:100%; margin:8px 0 28px; }
th, td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; } td.num { text-align:right; font-variant-numeric:tabular-nums; }
.pass { color:var(--pass); background:var(--pass-bg); } .fail { color:var(--fail); background:var(--fail-bg); }
.badge { border-radius:6px; padding:1px 6px; font-weight:600; font-size:12px; }
"""


def summarize(rows: list[dict]) -> list[dict]:
    """Per (suite, metric): case count, average score, pass rate."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r['suite'], r['metric'])].append(r)
    return [
        {
            'suite': suite,
            'metric': metric,
            'n': len(rs),
            'avg_score': sum(r['score'] or 0.0 for r in rs) / len(rs),
            'pass_rate': sum(bool(r['success']) for r in rs) / len(rs),
        }
        for (suite, metric), rs in sorted(groups.items())
    ]


def _badge(success: bool) -> str:
    return f'<span class="badge {"pass" if success else "fail"}">{"PASS" if success else "FAIL"}</span>'


def render_html(rows: list[dict], title: str = 'AI Quality Evaluation Report') -> str:
    total = len(rows)
    passed = sum(bool(r['success']) for r in rows)
    failed_rows = [r for r in rows if not r['success']]
    summary_rows = ''.join(
        f'<tr><td>{escape(s["suite"])}</td><td>{escape(s["metric"])}</td><td class="num">{s["n"]}</td>'
        f'<td class="num">{s["avg_score"]:.2f}</td><td class="num">{s["pass_rate"]:.0%}</td></tr>'
        for s in summarize(rows)
    )
    detail_rows = ''.join(
        f'<tr><td>{escape(r["suite"])}</td><td>{escape(r["case_id"])}</td><td>{escape(r["metric"])}</td>'
        f'<td class="num">{(r["score"] or 0.0):.2f}</td><td class="num">{"" if r["threshold"] is None else r["threshold"]}</td>'
        f'<td>{_badge(r["success"])}</td><td>{escape(r.get("reason") or "")}</td></tr>'
        for r in sorted(rows, key=lambda r: (bool(r['success']), r['suite'], r['case_id']))
    )
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title><style>{CSS}</style></head>
<body>
<h1>{escape(title)}</h1>
<div class="muted">Generated {generated} · judge model {escape(config.JUDGE_MODEL)}</div>
<div class="tiles">
  <div class="tile"><b>{total}</b>metric checks</div>
  <div class="tile"><b>{passed / total if total else 0:.0%}</b>pass rate</div>
  <div class="tile"><b>{len(failed_rows)}</b>failures</div>
  <div class="tile"><b>{len({r["case_id"] for r in rows})}</b>test cases</div>
</div>
<h2>Summary by metric</h2>
<div class="wrap"><table><tr><th>Suite</th><th>Metric</th><th>Cases</th><th>Avg score</th><th>Pass rate</th></tr>{summary_rows}</table></div>
<h2>All results (failures first)</h2>
<div class="wrap"><table><tr><th>Suite</th><th>Case</th><th>Metric</th><th>Score</th><th>Threshold</th><th>Result</th><th>Judge reason</th></tr>{detail_rows}</table></div>
</body></html>
"""


def load_rows(reports_dir: Path = REPORTS_DIR) -> list[dict]:
    rows = []
    for path in sorted(reports_dir.glob('*_results.json')):
        rows.extend(json.loads(path.read_text()))
    return rows


def write_report(rows: list[dict], path: Path = REPORTS_DIR / 'eval_report.html') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(rows))
    return path


if __name__ == '__main__':
    print(f'Wrote {write_report(load_rows())}')
