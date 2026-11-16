"""Module 6: Hybrid Search (BM25 Keyword + Vector Cosine Similarity + RRF).

Blends lexical search (Okapi BM25) with semantic vector search using
Reciprocal Rank Fusion (`RRF-K60`), exposing the exact rank and score from
each retriever so learners can see why Hybrid Search beats either alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from .duckdb_store import DuckDBVectorStore


@dataclass
class HybridSearchResult:
    """Detailed rank and score breakdown for a single chunk in Hybrid Search."""

    chunk_id: int
    doc_source: str
    section_title: str
    text: str
    vector_score: float
    vector_rank: int
    bm25_score: float
    bm25_rank: int
    rrf_score: float
    final_rank: int


class HybridSearchEngine:
    """Combines DuckDB vector similarity and BM25 using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, store: DuckDBVectorStore) -> None:
        self.store = store

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> list[HybridSearchResult]:
        """Execute Hybrid Search with weighted Reciprocal Rank Fusion (`RRF-K60`).

        Args:
            query: User search query.
            top_k: Number of fused results to return.
            alpha: Weight in [0.0, 1.0] for Vector Search (`1 - alpha` goes to BM25).
                - `alpha = 1.0`: Pure Semantic Vector Search
                - `alpha = 0.0`: Pure BM25 Keyword Search
                - `alpha = 0.5`: Balanced Hybrid RRF
            rrf_k: Smoothing constant `k` in `1 / (k + rank)` (default 60).
        """
        total_chunks = max(1, self.store.count())
        alpha = max(0.0, min(1.0, float(alpha)))
        rrf_k = max(1, int(rrf_k))

        vec_results = self.store.search_vector(query, top_k=total_chunks)
        bm25_results = self.store.search_bm25(query, top_k=total_chunks)

        vec_map = {
            rec.chunk_id: (rank_idx + 1, rec.score, rec)
            for rank_idx, rec in enumerate(vec_results)
        }
        bm25_map = {
            rec.chunk_id: (rank_idx + 1, rec.score, rec)
            for rank_idx, rec in enumerate(bm25_results)
        }

        all_ids = sorted(set(vec_map.keys()) | set(bm25_map.keys()))
        fused: list[HybridSearchResult] = []

        for cid in all_ids:
            v_rank, v_score, rec = vec_map[cid]
            b_rank, b_score, _ = bm25_map[cid]

            # If BM25 score is 0.0 (zero lexical overlap), assign a tail rank penalty
            effective_b_rank = b_rank if b_score > 0.0 else total_chunks + 50

            rrf_val = (alpha / (rrf_k + v_rank)) + (
                (1.0 - alpha) / (rrf_k + effective_b_rank)
            )
            fused.append(
                HybridSearchResult(
                    chunk_id=cid,
                    doc_source=rec.doc_source,
                    section_title=rec.section_title,
                    text=rec.text,
                    vector_score=round(v_score, 4),
                    vector_rank=v_rank,
                    bm25_score=round(b_score, 4),
                    bm25_rank=b_rank,
                    rrf_score=round(rrf_val, 6),
                    final_rank=0,
                )
            )

        # Sort by RRF score descending, breaking ties with vector_score + bm25_score
        fused.sort(
            key=lambda item: (
                item.rrf_score,
                item.vector_score * alpha + item.bm25_score * (1.0 - alpha),
            ),
            reverse=True,
        )

        top_results = fused[:top_k]
        for idx, item in enumerate(top_results):
            item.final_rank = idx + 1
        return top_results
