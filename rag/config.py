"""Central settings for the RAG pipeline and evals. Override any value via environment variables or .env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / 'data' / 'docs'
CHROMA_DIR = Path(os.getenv('CHROMA_DIR', ROOT / '.chroma'))
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'policy_docs_local')

# Embeddings run locally (Chroma's default all-MiniLM-L6-v2 ONNX model): free, no API key.
# Claude answers the questions; a cheaper Claude model judges them (DeepEval, RAGAS).
GENERATION_MODEL = os.getenv('GENERATION_MODEL', 'claude-opus-5-5')
JUDGE_MODEL = os.getenv('JUDGE_MODEL', 'claude-haiku-5-5')

TOP_K = int(os.getenv('TOP_K', '3'))
