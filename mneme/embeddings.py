"""Sentence-transformer embeddings for Mneme.

Uses ``all-MiniLM-L6-v2`` (384-dim) loaded lazily and cached in-process.

The model is ~80MB and is downloaded from HuggingFace on first use; subsequent
runs read it from the local cache. Set ``HF_HOME`` or ``TRANSFORMERS_CACHE``
to control where it lives.
"""

from functools import lru_cache

EMBEDDING_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def embed(text: str) -> list[float]:
    """Embed a single string into a 384-dim float vector."""
    model = _get_model()
    vec = model.encode(text or "", normalize_embeddings=True)
    return [float(v) for v in vec]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returned vectors are L2-normalized."""
    model = _get_model()
    if not texts:
        return []
    matrix = model.encode(
        [t or "" for t in texts],
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return [[float(v) for v in row] for row in matrix]


def cosine(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity. Assumes inputs are already normalized → dot product."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    return float(sum(a * b for a, b in zip(vec_a, vec_b)))
