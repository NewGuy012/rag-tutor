"""Pure-Python RAG from scratch (inspired by ngxson/make-your-own-rag)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def dot_product(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute the dot product of two equal-length vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch (ERR-VEC-768): {len(vec_a)} vs {len(vec_b)}"
        )
    return sum(a * b for a, b in zip(vec_a, vec_b))


def vector_norm(vec: list[float]) -> float:
    """Compute the Euclidean (L2) magnitude of a vector."""
    return math.sqrt(sum(x * x for x in vec))


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors in [-1.0, 1.0]."""
    norm_a = vector_norm(vec_a)
    norm_b = vector_norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product(vec_a, vec_b) / (norm_a * norm_b)


@dataclass
class SimpleVectorDB:
    """Minimal in-memory vector database backed by a plain Python list."""

    entries: list[tuple[str, list[float]]] = field(default_factory=list)

    def add_chunk(self, chunk: str, embedding: list[float]) -> None:
        cleaned = chunk.strip()
        if cleaned:
            self.entries.append((cleaned, embedding))

    def index_chunks(
        self,
        chunks: list[str],
        embed_fn: Callable[[str], list[float]],
    ) -> int:
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
        top_n = max(1, int(top_n))
        similarities = [
            (chunk, cosine_similarity(query_embedding, embedding))
            for chunk, embedding in self.entries
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]


def load_ev_facts(path: str | Path | None = None) -> list[str]:
    """Load the ev-facts.txt dataset (one electric vehicle fact per line)."""
    file_path = (
        Path(__file__).resolve().parents[2] / "data" / "ev-facts.txt"
        if path is None
        else Path(path)
    )
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_rag_prompt(
    query: str,
    retrieved_chunks: list[tuple[str, float]],
) -> tuple[str, str]:
    """Construct the grounded system prompt and user prompt."""
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
