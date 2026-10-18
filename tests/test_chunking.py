"""Offline checks for the RAG chunker (no API key needed)."""
from rag.chunking import chunk_markdown, load_chunks
from rag.config import DOCS_DIR

DOC = """# Refund Policy

## Refund window
Full refund within 30 days.

## Eligibility
Gift cards are non-refundable.
"""


def test_splits_on_section_headings():
    chunks = chunk_markdown(DOC, source='refund_policy')
    assert [c.section for c in chunks] == ['Refund window', 'Eligibility']
    assert [c.id for c in chunks] == ['refund_policy-0', 'refund_policy-1']


def test_chunk_text_carries_title_and_section():
    chunk = chunk_markdown(DOC, source='refund_policy')[0]
    assert chunk.text == 'Refund Policy - Refund window: Full refund within 30 days.'


def test_every_doc_produces_unique_chunks():
    chunks = load_chunks(DOCS_DIR)
    assert {c.source for c in chunks} == {p.stem for p in DOCS_DIR.glob('*.md')}
    assert len({c.id for c in chunks}) == len(chunks)
