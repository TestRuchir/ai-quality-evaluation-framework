import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    """DeepEval evals call OpenAI; skip them cleanly when no key is configured (e.g. forks in CI)."""
    if os.getenv('ANTHROPIC_API_KEY'):
        return
    skip = pytest.mark.skip(reason='ANTHROPIC_API_KEY not set; LLM evals skipped')
    evals_dir = Path(__file__).parent
    for item in items:
        if evals_dir in Path(item.fspath).parents:
            item.add_marker(skip)
