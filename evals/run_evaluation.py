"""Batch evaluation outside pytest: runs every DeepEval dataset through the RAG pipeline, agent, and chatbot;
scores them with DeepEval (quality, safety, agent, conversational) and RAGAS; writes JSON results, an HTML
report with change-vs-baseline, and a history entry; and prints a per-metric summary.

Usage:  python -m evals.run_evaluation [--skip-ragas] [--skip-safety] [--skip-agent] [--skip-chatbot]
Then:   python -m evals.regression check
"""
import json
import sys

from deepeval import evaluate
from deepeval.test_case import LLMTestCase

from evals.datasets import golden_id
from evals.metrics import generation_metrics, refusal_metric, retrieval_metrics
from evals.regression import append_history, compare, load_baseline, summary_by_key
from evals.report import REPORTS_DIR, summarize, write_report
from evals.test_rag_deepeval import IN_SCOPE, OUT_OF_SCOPE, build_test_case, run_pipeline


def deepeval_rows(suite: str, test_cases, metrics) -> list[dict]:
    """Every test case must carry `name` (the golden id) so results map back to dataset rows."""
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return [
        {
            'suite': suite,
            'case_id': test_result.name,
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
    from evals.ragas_metrics import THRESHOLDS, build_metrics, score

    metrics = build_metrics()
    rows = []
    for g in IN_SCOPE:
        result = run_pipeline(g.input)
        for name, value in score(metrics, g.input, result.answer, result.retrieval_context, g.expected_output).items():
            rows.append({
                'suite': 'rag-ragas', 'case_id': golden_id(g), 'metric': f'ragas_{name}', 'score': value,
                'threshold': THRESHOLDS[name], 'success': value >= THRESHOLDS[name], 'reason': None,
            })
    return rows


def safety_rows() -> list[dict]:
    from evals.safety_metrics import injection_resistance_metric, safety_metrics
    from evals.test_safety_deepeval import ATTACKS, CASES, agent_answer

    def case(g, output):
        return LLMTestCase(input=g.input, actual_output=output, name=golden_id(g))

    return (
        deepeval_rows('safety-rag', [case(g, run_pipeline(g.input).answer) for g in ATTACKS], [injection_resistance_metric()])
        + deepeval_rows('safety-agent', [case(g, agent_answer(g.input)) for g in CASES],
                        safety_metrics() + [injection_resistance_metric()])
    )


def agent_rows() -> list[dict]:
    from evals.test_agent_live import TASKS, agent_metrics, build_agent_test_case, run

    return deepeval_rows('agent', [build_agent_test_case(t, run(t['id'])) for t in TASKS], agent_metrics())


def chatbot_rows() -> list[dict]:
    from evals.test_chatbot_deepeval import CONVERSATIONS, build_conversational_test_case, conversation_metrics

    return deepeval_rows('chatbot', [build_conversational_test_case(g) for g in CONVERSATIONS], conversation_metrics())


def save(name: str, rows: list[dict]) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / f'{name}_results.json').write_text(json.dumps(rows, indent=2))


OPTIONAL_SUITES = {'ragas': ragas_rows, 'safety': safety_rows, 'agent': agent_rows, 'chatbot': chatbot_rows}


def main(argv: list[str]):
    rows = deepeval_rows('rag-deepeval', [build_test_case(g) for g in IN_SCOPE], retrieval_metrics() + generation_metrics())
    rows += deepeval_rows('rag-refusal', [build_test_case(g) for g in OUT_OF_SCOPE], [refusal_metric()])
    save('deepeval', rows)
    for name, run_suite in OPTIONAL_SUITES.items():
        if f'--skip-{name}' not in argv:
            suite_rows = run_suite()
            save(name, suite_rows)
            rows += suite_rows

    current = summary_by_key(rows)
    baseline = load_baseline()
    deltas = {c['key']: c for c in compare(current, baseline)} if baseline else None
    append_history(current)

    print(f'\n{"Suite":<16}{"Metric":<32}{"Avg score":>10}{"Pass rate":>12}')
    for s in summarize(rows):
        print(f'{s["suite"]:<16}{s["metric"]:<32}{s["avg_score"]:>10.2f}{s["pass_rate"]:>11.0%}')
    print(f'\nHTML report: {write_report(rows, deltas=deltas)}')
    print('Regression check: python -m evals.regression check')


if __name__ == '__main__':
    main(sys.argv[1:])
