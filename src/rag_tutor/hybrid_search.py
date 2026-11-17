"""Hybrid Search: BM25 + Vector Similarity with RRF & Convex Score Fusion."""

from __future__ import annotations

from dataclasses import dataclass

from .duckdb_store import DuckDBVectorStore


def _min_max_normalize(scores: dict[int, float]) -> dict[int, float]:
    """Scale raw scores into [0.0, 1.0] via Min-Max normalization."""
    if not scores:
        return {}
    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return {k: (1.0 if hi > 0.0 and v > 0.0 else 0.0) for k, v in scores.items()}
    span = hi - lo
    return {k: (v - lo) / span for k, v in scores.items()}


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
    norm_vector_score: float = 0.0
    norm_bm25_score: float = 0.0
    convex_score: float = 0.0
    fusion_method: str = "rrf"
    fusion_score: float = 0.0
    department: str = "general"
    updated_year: int = 2026


class HybridSearchEngine:
    """Combines DuckDB vector similarity and BM25 using RRF or Convex Combination."""

    def __init__(self, store: DuckDBVectorStore) -> None:
        self.store = store

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        rrf_k: int = 60,
        fusion_method: str = "rrf",
        department: str | None = None,
        min_year: int | None = None,
        query_vec: list[float] | None = None,
    ) -> list[HybridSearchResult]:
        """Execute Hybrid Search with RRF (`rrf`) or Normalized Convex Combination (`convex`)."""
        if fusion_method not in ("rrf", "convex"):
            raise ValueError(f"Unsupported fusion_method {fusion_method!r}; expected 'rrf' or 'convex'.")

        total_chunks = self.store.count()
        if total_chunks == 0:
            return []

        top_k = max(1, int(top_k))
        alpha = max(0.0, min(1.0, float(alpha)))
        rrf_k = max(1, int(rrf_k))

        vec_results = self.store.search_vector(
            query=query,
            top_k=total_chunks,
            department=department,
            min_year=min_year,
            query_vec=query_vec,
        )
        bm25_results = self.store.search_bm25(
            query=query,
            top_k=total_chunks,
            department=department,
            min_year=min_year,
        )
        if not vec_results:
            return []

        pool_size = len(vec_results)
        vec_map = {
            rec.chunk_id: (rank_idx + 1, rec.score, rec)
            for rank_idx, rec in enumerate(vec_results)
        }
        bm25_map = {
            rec.chunk_id: (rank_idx + 1, rec.score)
            for rank_idx, rec in enumerate(bm25_results)
        }

        norm_vec = _min_max_normalize({cid: v[1] for cid, v in vec_map.items()})
        norm_bm25 = _min_max_normalize(
            {cid: bm25_map.get(cid, (pool_size, 0.0))[1] for cid in vec_map}
        )

        fused: list[HybridSearchResult] = []
        for cid, (v_rank, v_score, rec) in vec_map.items():
            b_rank, b_score = bm25_map.get(cid, (pool_size, 0.0))
            effective_b_rank = b_rank if b_score > 0.0 else pool_size + 50
            rrf_val = (alpha / (rrf_k + v_rank)) + (
                (1.0 - alpha) / (rrf_k + effective_b_rank)
            )
            nv = norm_vec.get(cid, 0.0)
            nb = norm_bm25.get(cid, 0.0)
            convex_val = (alpha * nv) + ((1.0 - alpha) * nb)
            active_val = rrf_val if fusion_method == "rrf" else convex_val

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
                    norm_vector_score=round(nv, 4),
                    norm_bm25_score=round(nb, 4),
                    convex_score=round(convex_val, 6),
                    fusion_method=fusion_method,
                    fusion_score=round(active_val, 6),
                    department=rec.department,
                    updated_year=rec.updated_year,
                )
            )

        fused.sort(
            key=lambda item: (
                item.fusion_score,
                item.vector_score * alpha + item.bm25_score * (1.0 - alpha),
            ),
            reverse=True,
        )

        top_results = fused[:top_k]
        for idx, item in enumerate(top_results):
            item.final_rank = idx + 1
        return top_results
