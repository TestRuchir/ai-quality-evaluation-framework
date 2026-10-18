"""Section-based markdown chunking. Pure Python so it can be unit-tested offline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    section: str
    text: str


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """Split a markdown doc on `## ` headings; each chunk keeps the doc title and section heading for context."""
    title = source
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if line.startswith('# '):
            title = line[2:].strip()
        elif line.startswith('## '):
            sections.append((line[3:].strip(), []))
        elif line.strip() and sections:
            sections[-1][1].append(line.strip())

    return [
        Chunk(
            id=f'{source}-{i}',
            source=source,
            section=heading,
            text=f'{title} - {heading}: ' + ' '.join(body),
        )
        for i, (heading, body) in enumerate(sections)
        if body
    ]


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob('*.md')):
        chunks.extend(chunk_markdown(path.read_text(), source=path.stem))
    return chunks
