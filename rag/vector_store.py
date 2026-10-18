"""ChromaDB vector store with local embeddings: ingestion and top-k retrieval."""
from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from rag import config
from rag.chunking import load_chunks


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    source: str
    text: str
    distance: float


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),  # local; downloads ~80MB model on first use
        metadata={'hnsw:space': 'cosine'},
    )


def ingest(rebuild: bool = False) -> int:
    """Chunk all docs and upsert them into Chroma. Returns the number of chunks indexed."""
    if rebuild:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:  # collection did not exist yet
            pass
    collection = get_collection()
    chunks = load_chunks(config.DOCS_DIR)
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{'source': c.source, 'section': c.section} for c in chunks],
    )
    return len(chunks)


def retrieve(question: str, top_k: int = config.TOP_K) -> list[RetrievedChunk]:
    collection = get_collection()
    if collection.count() == 0:
        ingest()
    res = collection.query(query_texts=[question], n_results=top_k)
    return [
        RetrievedChunk(id=id_, source=meta['source'], text=doc, distance=dist)
        for id_, doc, meta, dist in zip(
            res['ids'][0], res['documents'][0], res['metadatas'][0], res['distances'][0]
        )
    ]


if __name__ == '__main__':
    print(f'Indexed {ingest(rebuild=True)} chunks into {config.CHROMA_DIR}')
