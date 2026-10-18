"""Golden datasets as DeepEval EvaluationDatasets.

The JSON files in datasets/ stay the source of truth (easy to diff and review); this module loads them into
DeepEval `Golden` / `ConversationalGolden` objects so tests and batch runs use DeepEval's dataset API.
Project-specific fields (id, category, expected_sources) travel in `additional_metadata`.
"""
from __future__ import annotations

import json

from deepeval.dataset import ConversationalGolden, EvaluationDataset, Golden

from rag import config

DATASETS_DIR = config.ROOT / 'datasets'


def _read(name: str) -> list[dict]:
    return json.loads((DATASETS_DIR / name).read_text())


def load_rag_dataset(category: str | None = None) -> EvaluationDataset:
    goldens = [
        Golden(
            input=g['input'],
            expected_output=g['expected_output'],
            context=g['context'] or None,
            name=g['id'],
            additional_metadata={'id': g['id'], 'category': g['category'], 'expected_sources': g['expected_sources']},
        )
        for g in _read('rag_golden.json')
        if category is None or g['category'] == category
    ]
    return EvaluationDataset(goldens=goldens)


def load_safety_dataset() -> EvaluationDataset:
    goldens = [
        Golden(
            input=c['input'],
            name=c['id'],
            additional_metadata={'id': c['id'], 'category': c['category'], 'should_block': c['should_block']},
        )
        for c in _read('safety_golden.json')
    ]
    return EvaluationDataset(goldens=goldens)


def load_conversation_dataset() -> EvaluationDataset:
    goldens = [
        ConversationalGolden(
            scenario=c['scenario'],
            expected_outcome=c['expected_outcome'],
            name=c['id'],
            additional_metadata={'id': c['id'], 'user_turns': c['user_turns'], 'must_mention': c['must_mention']},
        )
        for c in _read('chatbot_conversations.json')
    ]
    return EvaluationDataset(goldens=goldens)


def golden_id(golden) -> str:
    return golden.additional_metadata['id']
