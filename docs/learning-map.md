# Learning-to-Portfolio Map

| Learned topic | Demonstration in repository | Status |
|---|---|---|
| Golden datasets and benchmarking | `datasets/golden_dataset.json`, `test_llm_scoring.py` | Implemented demo |
| Deterministic evaluation and aggregation | `test_llm_scoring.py` | Implemented demo |
| Retrieval precision and recall | `test_rag_retrieval.py` | Implemented demo |
| White-box agent tool selection and argument validation | `test_agent_tools.py` | Implemented mock |
| Multi-turn chatbot evaluation | `test_chatbot_multiturn.py` | Implemented mock |
| End-to-end RAG (chunking, embeddings, vector DB, grounded generation) | `rag/` | Implemented demo |
| DeepEval LLMTestCase, RAG metrics, GEval, thresholds | `evals/test_rag_deepeval.py`, `evals/metrics.py` | Implemented demo |
| RAGAS faithfulness, answer relevancy, context precision/recall | `evals/ragas_metrics.py`, `evals/test_rag_ragas.py` | Implemented demo |
| LangGraph agent tools, guardrails, tracing | `agent/`, `tests/test_langgraph_agent.py`, `evals/test_agent_live.py` | Implemented demo |
| Prompt-injection test suite | `datasets/safety_golden.json`, `tests/test_guardrails.py` | Implemented demo |
| Model-based safety evaluation (toxicity, bias, PII, injection) | `evals/safety_metrics.py`, `evals/test_safety_deepeval.py` | Implemented demo |
| LLM-as-judge reliability via human review (Cohen's kappa) | `evals/human_review.py` | Implemented demo |
| HTML reporting | `evals/report.py` | Implemented demo |
| DeepEval EvaluationDataset / Golden / ConversationalGolden | `evals/datasets.py` | Implemented demo |
| Agent task completion and tool correctness (DeepEval) | `evals/test_agent_live.py`, `tests/test_deepeval_offline.py` | Implemented demo |
| Live multi-turn chatbot evaluation (ConversationalTestCase) | `chatbot/`, `evals/test_chatbot_deepeval.py`, `tests/test_claude_chatbot.py` | Implemented demo |
| Regression tracking against a baseline | `evals/regression.py`, `tests/test_regression.py` | Implemented demo |

**Important:** These are portfolio exercises; do not present them as client production code.
