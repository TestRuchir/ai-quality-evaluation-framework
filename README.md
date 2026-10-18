# AI Quality Evaluation Framework

A Python-first, **educational portfolio** demonstrating quality engineering for LLMs, RAG, chatbots, and tool-using agents. Built from hands-on learning in Pytest, DeepEval/RAGAS concepts, golden datasets, white-box tracing, black-box evaluation, and CI quality gates.

> **Scope:** Synthetic data. The tests in `tests/` are deterministic and run without an API key. The end-to-end RAG system in `rag/` and its DeepEval evaluation in `evals/` call Claude and need `ANTHROPIC_API_KEY` (from console.anthropic.com; API usage is billed separately from a Claude.ai subscription). Embeddings run locally, so building the index and retrieval checks are free.

## What is implemented
- End-to-end RAG system (`rag/`): markdown chunking → local embeddings (all-MiniLM-L6-v2) → ChromaDB retrieval → grounded Claude answer
- DeepEval evaluation of that RAG system (`evals/`) against a golden dataset (`datasets/rag_golden.json`)
- Golden-dataset checks for answer correctness and scoring aggregation
- Deterministic retrieval checks: hit rate, precision, recall, and groundedness checks for a small demo corpus
- Agent tool selection and argument correctness with inspectable traces
- Multi-turn chatbot memory and isolation checks: a mock bot, plus a real Claude chatbot (`chatbot/`) scored with DeepEval conversational metrics (knowledge retention, turn relevancy, role adherence, conversation completeness)
- DeepEval `EvaluationDataset` / `Golden` / `ConversationalGolden` loaded from the JSON golden datasets (`evals/datasets.py`)
- Agent task-level metrics: DeepEval `ToolCorrectnessMetric` and `TaskCompletionMetric` on the live agent's trace (`evals/test_agent_live.py`)
- Regression tracking: committed score baseline, CI gate on metric drops, run history, and change-vs-baseline in the HTML report (`evals/regression.py`)
- RAGAS evaluation (faithfulness, answer relevancy, context precision/recall) alongside DeepEval (`evals/ragas_metrics.py`)
- LangGraph support agent (`agent/`): tools (policy search, order lookup with per-customer authorization, safe calculator), input/output guardrails, step limit, and white-box trace capture, tested offline with a scripted chat model
- Prompt-injection / jailbreak test suite (`datasets/safety_golden.json`) for the rule-based guard, including false-positive checks
- Model-based safety evaluation: DeepEval Toxicity, Bias, PII leakage, and a GEval injection-resistance judge (`evals/test_safety_deepeval.py`)
- Human review of LLM-judge verdicts with Cohen's kappa agreement (`evals/human_review.py`)
- HTML eval report (`evals/report.py`), uploaded as a CI artifact
- Pytest + GitHub Actions CI

## Run locally
```bash
python3.11 -m venv .venv   # Python 3.10+
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m pytest -v
```

## RAG system + DeepEval evals (Python 3.10+)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-rag.txt
cp .env.example .env            # add your ANTHROPIC_API_KEY

python -m rag.vector_store      # build the Chroma index from data/docs/
python -m rag.pipeline "Does the warranty cover water damage?"

deepeval test run evals/        # pytest-style quality gate (fails if any metric is below threshold)
python -m evals.run_evaluation  # DeepEval + RAGAS + safety batch run -> reports/*.json + reports/eval_report.html

python -m agent.graph "Can I still return order 1002?"   # LangGraph agent with printed tool trace
python -m chatbot.bot                                       # interactive multi-turn Claude chatbot

python -m evals.regression update   # promote this run's scores to baselines/eval_baseline.json (commit it)
python -m evals.regression check    # exit 1 if any metric regressed vs the baseline (runs in CI)

python -m evals.human_review export      # reports/human_review.csv: fill in human_pass (yes/no)
python -m evals.human_review agreement   # judge vs human agreement + Cohen's kappa per metric
```

```text
question ─► retriever (Chroma, top-k) ─► retrieval_context ─► generator (Claude) ─► answer
                     │                                                 │
       ContextualPrecision / Recall / Relevancy       Faithfulness / AnswerRelevancy /
       + deterministic hit-rate                        Hallucination / GEval Correctness
                                                       GEval Refusal (out-of-scope + prompt injection)
```

| DeepEval metric | Component | Uses | Question it answers |
|---|---|---|---|
| `ContextualPrecisionMetric` | Retriever | input, expected_output, retrieval_context | Are relevant chunks ranked above irrelevant ones? |
| `ContextualRecallMetric` | Retriever | expected_output, retrieval_context | Does the retrieved context contain everything needed for the ideal answer? |
| `ContextualRelevancyMetric` | Retriever | input, retrieval_context | How much of the retrieved text is relevant to the question? |
| `FaithfulnessMetric` | Generator | actual_output, retrieval_context | Are all claims in the answer supported by the retrieved context? |
| `AnswerRelevancyMetric` | Generator | input, actual_output | Does the answer address the question? |
| `HallucinationMetric` | Generator | actual_output, context (ground truth) | Does the answer contradict known facts? (lower is better) |
| `GEval` Correctness | Generator | input, actual_output, expected_output | Does the answer match the expected facts? |
| `GEval` Refusal | Guardrail | input, actual_output | Does the bot decline unanswerable / injected questions instead of inventing answers? |

Answers use `claude-opus-5-5` and the LLM judge uses `claude-haiku-5-5` (override with `GENERATION_MODEL` / `JUDGE_MODEL`; e.g. `GENERATION_MODEL=claude-haiku-5-5` for the cheapest runs). Thresholds are set in `evals/metrics.py`. Retrieval and generation are tested separately, so a failing test shows which component regressed.

## Architecture
```text
Synthetic golden datasets
          |
          v
Pytest evaluation harness
   |       |       |
  LLM     RAG    Agent + Chatbot
   |       |       |
   +-------+-------+
           |
  Assertions + aggregate scores
           |
     GitHub Actions
```

## Metrics and evaluation methods
| Area | Example metrics | Method |
|---|---|---|
| LLM | Exact match, answer correctness | Deterministic |
| RAG | Retrieval precision/recall, groundedness | Deterministic demo + RAGAS (LLM judge) |
| Agent | Tool choice, argument correctness, step count, authorization | White-box trace assertions on a LangGraph agent (scripted LLM offline, real LLM in `evals/`) |
| Safety | Injection catch rate, false-positive rate, toxicity, bias, PII leakage, injection resistance | Rule-based guard tests + DeepEval LLM judge |
| Judge reliability | Raw agreement, Cohen's kappa | Human review CSV |
| Regression | Score / pass-rate change vs baseline | Committed baseline + tolerance-based CI gate + run history |
| Chatbot | Context retention, session isolation, knowledge retention, role adherence | Multi-turn state tests (mock + fake client) and DeepEval conversational metrics on the live Claude bot |
| Live RAG | Contextual precision/recall/relevancy, faithfulness, answer relevancy, hallucination, GEval | DeepEval LLM-as-judge |

## Possible extensions
- Larger golden datasets and adversarial cases generated with DeepEval/RAGAS synthesizers
- LLM-based input classifier to complement the regex injection guard
- Trend charts from `reports/history.jsonl`

## Portfolio / interview talking points
- Why deterministic exact-match metrics are inadequate for open-ended LLM outputs
- How to distinguish retrieval failures from generation failures
- White-box vs black-box agent evaluation
- How to design golden datasets and apply thresholds in CI
- How to validate multi-turn conversations and tool authorization

## Ethics and confidentiality
This repository contains original demonstration code and synthetic examples only. It does not contain EY/client code, private data, credentials, or internal artifacts.
