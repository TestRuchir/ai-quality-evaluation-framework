"""Batch evaluation outside pytest: runs every golden through the RAG pipeline and the agent, scores them with
DeepEval (quality + safety) and RAGAS, prints a per-metric summary, and writes JSON results + an HTML report.

Usage:  python -m evals.run_evaluation [--skip-ragas] [--skip-safety]
"""
import json
import sys

from deepeval import evaluate

from evals.metrics import generation_metrics, refusal_metric, retrieval_metrics
from evals.ragas_metrics import THRESHOLDS as RAGAS_THRESHOLDS
from evals.report import REPORTS_DIR, summarize, write_report
from evals.test_rag_deepeval import IN_SCOPE, OUT_OF_SCOPE, build_test_case, run_pipeline


def deepeval_rows(suite: str, goldens: list[dict], test_cases, metrics) -> list[dict]:
    ids_by_input = {g['input']: g['id'] for g in goldens}
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return [
        {
            'suite': suite,
            'case_id': ids_by_input.get(test_result.input, test_result.name),
            'metric': m.name,
            'score': m.score,
            'threshold': m.threshold,
            'success': m.success,
            'reason': m.reason,
        }
        for test_result in result.test_results
        for m in test_result.metrics_data
    ]


def ragas_rows() -> list[dict]:
    from evals.ragas_metrics import build_metrics, score

    metrics = build_metrics()
    rows = []
    for g in IN_SCOPE:
        result = run_pipeline(g['input'])
        for name, value in score(metrics, g['input'], result.answer, result.retrieval_context, g['expected_output']).items():
            rows.append({
                'suite': 'rag-ragas', 'case_id': g['id'], 'metric': f'ragas_{name}', 'score': value,
                'threshold': RAGAS_THRESHOLDS[name], 'success': value >= RAGAS_THRESHOLDS[name], 'reason': None,
            })
    return rows


def safety_rows() -> list[dict]:
    from deepeval.test_case import LLMTestCase

    from evals.safety_metrics import injection_resistance_metric, safety_metrics
    from evals.test_safety_deepeval import ATTACKS, CASES, agent_answer

    rag_cases = [LLMTestCase(input=c['input'], actual_output=run_pipeline(c['input']).answer) for c in ATTACKS]
    agent_cases = [LLMTestCase(input=c['input'], actual_output=agent_answer(c['input'])) for c in CASES]
    return (
        deepeval_rows('safety-rag', ATTACKS, rag_cases, [injection_resistance_metric()])
        + deepeval_rows('safety-agent', CASES, agent_cases, safety_metrics() + [injection_resistance_metric()])
    )


def save(name: str, rows: list[dict]) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f'{name}_results.json').write_text(json.dumps(rows, indent=2))


def main(argv: list[str]):
    rows = deepeval_rows(
        'rag-deepeval', IN_SCOPE, [build_test_case(g) for g in IN_SCOPE], retrieval_metrics() + generation_metrics()
    ) + deepeval_rows('rag-refusal', OUT_OF_SCOPE, [build_test_case(g) for g in OUT_OF_SCOPE], [refusal_metric()])
    save('deepeval', rows)

    if '--skip-ragas' not in argv:
        ragas = ragas_rows()
        save('ragas', ragas)
        rows += ragas
    if '--skip-safety' not in argv:
        safety = safety_rows()
        save('safety', safety)
        rows += safety

    print(f'\n{"Suite":<16}{"Metric":<32}{"Avg score":>10}{"Pass rate":>12}')
    for s in summarize(rows):
        print(f'{s["suite"]:<16}{s["metric"]:<32}{s["avg_score"]:>10.2f}{s["pass_rate"]:>11.0%}')
    print(f'\nHTML report: {write_report(rows)}')


if __name__ == '__main__':
    main(sys.argv[1:])
