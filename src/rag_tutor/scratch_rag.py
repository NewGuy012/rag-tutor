"""Module 1-3: Pure-Python RAG from scratch (inspired by ngxson/make-your-own-rag).

This module intentionally uses zero external math or database libraries so
beginners can read every line of code and see that vector search is simply
dot products and vector magnitudes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute the dot product of two equal-length vectors in pure Python.

    Formula: sum(a * b for a, b in zip(vec_a, vec_b))
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch (ERR-VEC-768): {len(vec_a)} vs {len(vec_b)}"
        )
    return sum(a * b for a, b in zip(vec_a, vec_b))


def vector_norm(vec: list[float]) -> float:
    """Compute the Euclidean (L2) magnitude of a vector in pure Python.

    Formula: sqrt(sum(x^2 for x in vec))
    """
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute the cosine similarity between two vectors in [-1.0, 1.0].

    Matches the exact implementation from Xuan-Son Nguyen's
    'Code a simple RAG from scratch' tutorial.
    """
    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product(vec_a, vec_b) / (norm_a * norm_b)


@dataclass
class SimpleVectorDB:
    """A minimal in-memory vector database backed by a plain Python list.

    Each entry is a tuple of (chunk_text, embedding_vector).
    """

    entries: list[tuple[str, list[float]]] = field(default_factory=list)

    def add_chunk(self, chunk: str, embedding: list[float]) -> None:
        """Add a single text chunk and its embedding vector to the list."""
        cleaned = chunk.strip()
        if cleaned:
            self.entries.append((cleaned, embedding))

    def index_chunks(
        self,
        chunks: list[str],
        embed_fn: Callable[[str], list[float]],
    ) -> int:
        """Embed and store a list of text chunks. Returns total indexed count."""
        for chunk in chunks:
            cleaned = chunk.strip()
            if cleaned:
                self.add_chunk(cleaned, embed_fn(cleaned))
        return len(self.entries)

    def retrieve(
        self,
        query_embedding: list[float],
        top_n: int = 3,
    ) -> list[tuple[str, float]]:
        """Return the top_n (chunk, similarity_score) pairs sorted descending."""
        similarities: list[tuple[str, float]] = []
        for chunk, embedding in self.entries:
            sim = cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, sim))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]


def load_ev_facts(path: str | Path | None = None) -> list[str]:
    """Load the ev-facts.txt dataset (one electric vehicle fact per line)."""
    if path is None:
        path = Path(__file__).resolve().parents[2] / "data" / "ev-facts.txt"
    else:
        path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[tuple[str, float]],
) -> tuple[str, str]:
    """Construct the system instruction prompt and user prompt for grounded RAG.

    Follows the prompt design from ngxson's tutorial where the LLM is strictly
    instructed to rely only on the provided context chunks.
    """
    context_lines = "\n".join(
        f" - [Fact #{idx + 1} | similarity={score:.3f}] {chunk}"
        for idx, (chunk, score) in enumerate(retrieved_chunks)
    )
    system_prompt = (
        "You are a helpful, accurate educational assistant.\n"
        "Use ONLY the following pieces of context to answer the user's question. "
        "Do not make up any new information or extrapolate beyond the facts given:\n"
        f"{context_lines}"
    )
    return system_prompt, query
