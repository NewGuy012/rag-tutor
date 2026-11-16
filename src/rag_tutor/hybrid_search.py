"""Hybrid Search: BM25 Keyword + Vector Cosine Similarity + Reciprocal Rank Fusion (`RRF-K60`)."""

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
        """Execute Hybrid Search with weighted Reciprocal Rank Fusion (`RRF-K60`)."""
        total_chunks = self.store.count()
        if total_chunks == 0:
            return []

        top_k = max(1, int(top_k))
        alpha = max(0.0, min(1.0, float(alpha)))
        rrf_k = max(1, int(rrf_k))

        vec_results = self.store.search_vector(query, top_k=total_chunks)
        bm25_results = self.store.search_bm25(query, top_k=total_chunks)

        vec_map = {
            rec.chunk_id: (rank_idx + 1, rec.score, rec)
            for rank_idx, rec in enumerate(vec_results)
        }
        bm25_map = {
            rec.chunk_id: (rank_idx + 1, rec.score)
            for rank_idx, rec in enumerate(bm25_results)
        }

        fused: list[HybridSearchResult] = []
        for cid, (v_rank, v_score, rec) in vec_map.items():
            b_rank, b_score = bm25_map.get(cid, (total_chunks, 0.0))
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
