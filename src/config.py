import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)
COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "cmu_technical_docs"
)

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

openai_client = OpenAI()

qdrant_client = QdrantClient(
    url=QDRANT_URL
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)