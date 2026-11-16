"""Advanced Graph, Hybrid, and Modular RAG Pipeline.

Implements the three stages of modern Hybrid & Modular RAG architectures:
1. Pre-Retrieval Query Expansion (`Multi-Query` and `HyDE`).
2. Parallel Retrieval (`Dense + Sparse`, `Vector + Knowledge Graph`, `Vector + SQL Filters`).
3. Score Fusion (`RRF` vs. `Convex Combination`) and Post-Retrieval `Cross-Encoder` Reranking.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Literal

from .chunker import Chunk
from .duckdb_store import DuckDBVectorStore
from .hybrid_search import HybridSearchEngine, HybridSearchResult
from .llm_backend import LlamaMacOSClient, get_default_client, get_default_embedder, tokenize
from .reranker import CrossEncoderReranker, RerankedCandidate

ExpansionMode = Literal["none", "multi_query", "hyde"]
FusionMethod = Literal["rrf", "convex"]

_EXPLICIT_TRIPLE_RE = re.compile(
    r"\(([A-Za-z0-9][A-Za-z0-9 ./_-]{1,40}),\s*([a-zA-Z_][a-zA-Z0-9 _-]{1,32}),\s*([A-Za-z0-9][A-Za-z0-9 ./_>-]{1,48})\)"
)

_CANONICAL_TRIPLES: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("DuckDB", "stores_vectors_as", "FLOAT[128] Arrays", ("duckdb", "float", "embedding", "table")),
    ("DuckDB", "computes_similarity_via", "array_cosine_similarity", ("duckdb", "array_cosine_similarity")),
    ("DuckDB", "supports_sparse_search", "Okapi BM25 (BM25-K1-1.2)", ("duckdb", "fts", "bm25", "bm25-k1-1.2")),
    ("DuckDB", "scales_ann_via", "HNSW (vss extension)", ("duckdb", "vss", "hnsw")),
    ("ERR-VEC-768", "caused_by", "Embedding Dimension Mismatch", ("err-vec-768", "dimension", "mismatch")),
    ("ERR-CHUNK-004", "caused_by", "Zero Vector Norm (<15 chars)", ("err-chunk-004", "zero", "norm")),
    ("Hybrid Search", "combines", "Dense Vectors + Sparse BM25", ("hybrid", "bm25", "vector")),
    ("RRF-K60", "merges_ranks_via", "Reciprocal Rank Fusion", ("rrf", "rrf-k60", "reciprocal")),
    ("CONVEX-ALPHA-70", "blends_scores_via", "Min-Max Normalized Linear Sum", ("convex", "linear", "alpha")),
    ("Cross-Encoder", "refines", "Top-50 Candidates -> Top-5", ("cross-encoder", "rerank", "top-50", "cohere")),
    ("STAGE2-RERANK-15to3", "prevents", "Lost in the Middle", ("stage2-rerank-15to3", "lost", "middle")),
    ("HyDE", "generates", "Hypothetical Answer Embedding", ("hyde", "hypothetical", "answer")),
    ("Multi-Query", "rewrites_into", "3-4 Query Variations", ("multi-query", "variations", "expansion")),
    ("GraphRAG", "traverses", "Knowledge Graph Triples", ("graphrag", "graph", "triples")),
    ("Qdrant", "supports_natively", "Dense + Sparse SPLADE + RRF", ("qdrant", "splade", "rrf")),
    ("Weaviate", "supports_natively", "HNSW + BM25F + Convex/RRF", ("weaviate", "bm25f", "relativescorefusion")),
    ("Pinecone", "supports_natively", "Hybrid Indexes + Rerankers", ("pinecone", "hybrid", "rerankers")),
    ("PROTO-RAG-101", "illustrates", "Ungrounded LLM Hallucination", ("proto-rag-101", "hallucination")),
    ("Lewis et al. (2020)", "introduced", "Retrieval-Augmented Generation", ("lewis", "2020", "rag")),
]


@dataclass(frozen=True)
class KnowledgeTriple:
    """A directed `(subject, predicate, obj)` relationship grounded in a corpus chunk."""

    subject: str
    predicate: str
    obj: str
    source_chunk_id: int
    doc_source: str
    section_title: str


@dataclass
class GraphSearchResult:
    """Result of 1-to-2 hop knowledge graph traversal for a query."""

    seed_entities: list[str]
    matched_triples: list[tuple[KnowledgeTriple, int]]
    chunk_scores: dict[int, float]


@dataclass
class QueryExpansionResult:
    """Expanded queries and optional HyDE hypothetical passage."""

    mode: ExpansionMode
    original_query: str
    expanded_queries: list[str]
    hypothetical_doc: str = ""
    used_llm: bool = False


@dataclass
class ModularRAGConfig:
    """Tunable knobs for each stage of the Modular RAG pipeline."""

    expansion_mode: ExpansionMode = "none"
    enable_dense_sparse: bool = True
    enable_graph: bool = True
    department: str | None = None
    min_year: int | None = None
    fusion_method: FusionMethod = "rrf"
    alpha: float = 0.5
    rrf_k: int = 60
    graph_weight: float = 0.35
    use_reranker: bool = True
    stage1_k: int = 12
    final_top_k: int = 4


@dataclass
class ModularCandidate:
    """A retrieved candidate enriched with multi-engine provenance."""

    chunk_id: int
    doc_source: str
    section_title: str
    text: str
    department: str
    updated_year: int
    base_fusion_score: float
    graph_bonus: float
    combined_score: float
    rerank_score: float | None
    sources: list[str]


@dataclass
class ModularRAGTrace:
    """Full execution trace across pre-retrieval, parallel retrieval, fusion, and refinement."""

    query: str
    config: ModularRAGConfig
    expansion: QueryExpansionResult
    graph_result: GraphSearchResult
    sql_filter_preview: str
    stage1_candidates: list[ModularCandidate]
    final_candidates: list[ModularCandidate]
    final_answer: str
    llm_model_used: str
    llm_backend_used: str
    latency_ms: float


class KnowledgeGraphStore:
    """Lightweight Knowledge Graph built from corpus chunks for Vector + Graph RAG."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks_by_id: dict[int, Chunk] = {c.chunk_id: c for c in chunks}
        self.triples: list[KnowledgeTriple] = []
        self._adj: dict[str, list[tuple[str, KnowledgeTriple]]] = defaultdict(list)
        self._build_from_chunks(chunks)

    def _find_best_chunk(self, chunks: list[Chunk], keywords: tuple[str, ...]) -> Chunk | None:
        if not chunks:
            return None
        kw_set = {k.lower() for k in keywords}
        best_chunk = chunks[0]
        best_hits = -1
        for ch in chunks:
            text_low = f"{ch.section_title} {ch.text}".lower()
            hits = sum(1 for kw in kw_set if kw in text_low)
            if hits > best_hits:
                best_hits = hits
                best_chunk = ch
        return best_chunk if best_hits > 0 else chunks[0]

    def _add_triple(self, triple: KnowledgeTriple, seen: set[tuple[str, str, str]]) -> None:
        key = (triple.subject.lower(), triple.predicate.lower(), triple.obj.lower())
        if key in seen:
            return
        seen.add(key)
        self.triples.append(triple)
        self._adj[triple.subject].append((triple.obj, triple))
        self._adj[triple.obj].append((triple.subject, triple))

    def _build_from_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        seen: set[tuple[str, str, str]] = set()
        for subj, pred, obj, kws in _CANONICAL_TRIPLES:
            ch = self._find_best_chunk(chunks, kws)
            if ch is not None:
                self._add_triple(
                    KnowledgeTriple(
                        subject=subj,
                        predicate=pred,
                        obj=obj,
                        source_chunk_id=ch.chunk_id,
                        doc_source=ch.doc_source,
                        section_title=ch.section_title,
                    ),
                    seen,
                )

        # Also extract explicit inline triples written as `(Subject, predicate, Object)` in text.
        for ch in chunks:
            for subj, pred, obj in _EXPLICIT_TRIPLE_RE.findall(ch.text):
                self._add_triple(
                    KnowledgeTriple(
                        subject=subj.strip(),
                        predicate=pred.strip().replace(" ", "_"),
                        obj=obj.strip(),
                        source_chunk_id=ch.chunk_id,
                        doc_source=ch.doc_source,
                        section_title=ch.section_title,
                    ),
                    seen,
                )

    def search(
        self,
        query: str,
        max_hops: int = 2,
        max_triples: int = 12,
    ) -> GraphSearchResult:
        """Find entity seeds matching `query` and traverse up to `max_hops` graph edges."""
        max_hops = max(1, min(3, int(max_hops)))
        q_lower = query.lower()
        q_tokens = set(tokenize(query, stem=True, drop_stopwords=True))

        entity_scores: list[tuple[float, str]] = []
        for entity in self._adj:
            e_lower = entity.lower()
            e_tokens = set(tokenize(entity, stem=True, drop_stopwords=True))
            overlap = len(q_tokens & e_tokens)
            exact_phrase = 2.5 if len(e_lower) >= 3 and e_lower in q_lower else 0.0
            score = exact_phrase + overlap
            if score > 0.0:
                entity_scores.append((score, entity))

        entity_scores.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        seed_entities = [ent for _, ent in entity_scores[:4]]
        if not seed_entities:
            return GraphSearchResult(seed_entities=[], matched_triples=[], chunk_scores={})

        visited_entities: dict[str, int] = {seed: 0 for seed in seed_entities}
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seed_entities)
        matched_triples: list[tuple[KnowledgeTriple, int]] = []
        seen_triples: set[tuple[str, str, str]] = set()
        raw_chunk_scores: dict[int, float] = defaultdict(float)

        while queue:
            curr_entity, dist = queue.popleft()
            if dist >= max_hops:
                continue
            next_hop = dist + 1
            for neighbor, triple in self._adj.get(curr_entity, []):
                t_key = (triple.subject, triple.predicate, triple.obj)
                if t_key not in seen_triples:
                    seen_triples.add(t_key)
                    matched_triples.append((triple, next_hop))
                    hop_weight = 1.0 if next_hop == 1 else 0.45
                    raw_chunk_scores[triple.source_chunk_id] += hop_weight
                if neighbor not in visited_entities:
                    visited_entities[neighbor] = next_hop
                    queue.append((neighbor, next_hop))

        matched_triples.sort(key=lambda item: (item[1], item[0].source_chunk_id))
        matched_triples = matched_triples[:max_triples]

        peak = max(raw_chunk_scores.values(), default=0.0)
        norm_chunk_scores = (
            {cid: round(sc / peak, 4) for cid, sc in raw_chunk_scores.items()}
            if peak > 0.0
            else {}
        )
        return GraphSearchResult(
            seed_entities=seed_entities,
            matched_triples=matched_triples,
            chunk_scores=norm_chunk_scores,
        )


class QueryExpander:
    """Pre-retrieval query expansion supporting Multi-Query and HyDE."""

    def __init__(self, llm_client: LlamaMacOSClient | None = None) -> None:
        self.llm_client = llm_client or get_default_client()

    def expand(
        self,
        query: str,
        mode: ExpansionMode = "none",
        model: str | None = None,
    ) -> QueryExpansionResult:
        """Expand `query` using `'none'`, `'multi_query'`, or `'hyde'`."""
        cleaned = query.strip()
        if mode == "none" or not cleaned:
            return QueryExpansionResult(
                mode="none",
                original_query=cleaned,
                expanded_queries=[cleaned] if cleaned else [],
            )

        if mode == "multi_query":
            return self._expand_multi_query(cleaned, model=model)
        if mode == "hyde":
            return self._expand_hyde(cleaned, model=model)
        raise ValueError(f"Unsupported expansion mode {mode!r}")

    def _expand_multi_query(
        self,
        query: str,
        model: str | None = None,
    ) -> QueryExpansionResult:
        used_llm = False
        variations: list[str] = []
        if model and model != "educational-fallback":
            res = self.llm_client.generate(
                system_prompt=(
                    "Rewrite the user question into 3 concise search query variations covering "
                    "different synonyms and exact technical terms. Return one query per line."
                ),
                user_prompt=query,
                model=model,
                max_tokens=160,
                retrieved_context=None,
            )
            if res.backend == "llama-macos":
                used_llm = True
                for line in res.content.splitlines():
                    item = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
                    if len(item) >= 8 and item.lower() != query.lower():
                        variations.append(item)

        if not variations:
            tokens = tokenize(query, stem=False, drop_stopwords=True)
            key_terms = " ".join(tokens[:6]) or query
            variations = [
                f"{key_terms} architecture workflow and pipeline stages",
                f"exact technical terms and configuration for {key_terms}",
                f"comparison and tuning best practices for {key_terms}",
            ]

        unique_queries = [query]
        for v in variations:
            if v not in unique_queries:
                unique_queries.append(v)
        return QueryExpansionResult(
            mode="multi_query",
            original_query=query,
            expanded_queries=unique_queries[:4],
            used_llm=used_llm,
        )

    def _expand_hyde(
        self,
        query: str,
        model: str | None = None,
    ) -> QueryExpansionResult:
        used_llm = False
        hypo = ""
        if model and model != "educational-fallback":
            res = self.llm_client.generate(
                system_prompt=(
                    "Write a concise, factual 2-sentence technical passage that directly answers "
                    "the user's question as if excerpted from an engineering reference manual."
                ),
                user_prompt=query,
                model=model,
                max_tokens=180,
                retrieved_context=None,
            )
            if res.backend == "llama-macos" and len(res.content.strip()) >= 20:
                used_llm = True
                hypo = res.content.strip()

        if not hypo:
            tokens = tokenize(query, stem=False, drop_stopwords=True)
            focus = ", ".join(tokens[:6]) if tokens else query
            hypo = (
                f"In production RAG architectures addressing {focus}, systems pair dense vector "
                "similarity search with lexical BM25 keyword scoring, optional Knowledge Graph "
                "traversal, and SQL metadata filtering. Candidate lists are merged via Reciprocal "
                "Rank Fusion (RRF-K60) or Min-Max normalized Convex Combination before a Stage-2 "
                "Cross-Encoder reranker refines the final top chunks for grounded LLM synthesis."
            )

        return QueryExpansionResult(
            mode="hyde",
            original_query=query,
            expanded_queries=[query],
            hypothetical_doc=hypo,
            used_llm=used_llm,
        )


class ModularRAGPipeline:
    """Composable Modular RAG orchestrator combining Expansion, Parallel Retrieval, Fusion, and Reranking."""

    def __init__(
        self,
        store: DuckDBVectorStore,
        chunks: list[Chunk],
        hybrid_engine: HybridSearchEngine | None = None,
        graph_store: KnowledgeGraphStore | None = None,
        reranker: CrossEncoderReranker | None = None,
        expander: QueryExpander | None = None,
        llm_client: LlamaMacOSClient | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.store = store
        self.chunks_by_id = {c.chunk_id: c for c in chunks}
        self.hybrid_engine = hybrid_engine or HybridSearchEngine(store)
        self.graph_store = graph_store or KnowledgeGraphStore(chunks)
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_client = llm_client or get_default_client()
        self.expander = expander or QueryExpander(self.llm_client)
        self.embed_fn = embed_fn or get_default_embedder().embed

    def _chunk_passes_filters(self, chunk: Chunk, config: ModularRAGConfig) -> bool:
        if config.department and config.department != "all" and chunk.department != config.department:
            return False
        if config.min_year is not None and chunk.updated_year < int(config.min_year):
            return False
        return True

    def run(
        self,
        query: str,
        config: ModularRAGConfig | None = None,
        model: str | None = None,
        enable_thinking: bool = False,
    ) -> ModularRAGTrace:
        """Execute the configured Modular RAG pipeline and return a full trace."""
        cfg = config or ModularRAGConfig()
        expansion = self.expander.expand(query, mode=cfg.expansion_mode, model=model)

        sql_preview, sql_params = self.store.build_filtered_vector_sql(
            department=cfg.department,
            min_year=cfg.min_year,
        )
        if sql_params:
            sql_preview = f"{sql_preview}\n-- Filter params: {sql_params!r}"

        hyde_vec = (
            self.embed_fn(expansion.hypothetical_doc)
            if expansion.mode == "hyde" and expansion.hypothetical_doc
            else None
        )

        # Stage 1 & 2: Dense + Sparse retrieval across expanded queries
        candidates_by_id: dict[int, ModularCandidate] = {}
        multi_query_rrf: dict[int, float] = defaultdict(float)

        if cfg.enable_dense_sparse:
            for q_idx, q_variant in enumerate(expansion.expanded_queries or [query]):
                hits = self.hybrid_engine.search(
                    query=q_variant,
                    top_k=cfg.stage1_k,
                    alpha=cfg.alpha,
                    rrf_k=cfg.rrf_k,
                    fusion_method=cfg.fusion_method,
                    department=cfg.department,
                    min_year=cfg.min_year,
                    query_vec=hyde_vec if q_idx == 0 else None,
                )
                for rank_pos, h in enumerate(hits):
                    # When multiple queries run, combine per-query rankings smoothly
                    if len(expansion.expanded_queries) > 1:
                        multi_query_rrf[h.chunk_id] += 1.0 / (cfg.rrf_k + rank_pos + 1)
                    existing = candidates_by_id.get(h.chunk_id)
                    if existing is None:
                        srcs = []
                        if cfg.alpha > 0.0:
                            srcs.append("HyDE-Dense" if hyde_vec is not None else "Dense")
                        if cfg.alpha < 1.0:
                            srcs.append("BM25")
                        if (cfg.department and cfg.department != "all") or cfg.min_year is not None:
                            srcs.append("SQL-Filter")
                        candidates_by_id[h.chunk_id] = ModularCandidate(
                            chunk_id=h.chunk_id,
                            doc_source=h.doc_source,
                            section_title=h.section_title,
                            text=h.text,
                            department=h.department,
                            updated_year=h.updated_year,
                            base_fusion_score=h.fusion_score,
                            graph_bonus=0.0,
                            combined_score=h.fusion_score,
                            rerank_score=None,
                            sources=srcs,
                        )

            if len(expansion.expanded_queries) > 1:
                for cid, cand in candidates_by_id.items():
                    cand.base_fusion_score = round(multi_query_rrf[cid], 6)
                    cand.combined_score = cand.base_fusion_score
                    if "Multi-Query" not in cand.sources:
                        cand.sources.append("Multi-Query")

        # Parallel Graph traversal (respecting SQL metadata filters if enabled)
        graph_result = (
            self.graph_store.search(query, max_hops=2)
            if cfg.enable_graph
            else GraphSearchResult(seed_entities=[], matched_triples=[], chunk_scores={})
        )

        if cfg.enable_graph and graph_result.chunk_scores:
            peak_base = max(
                (c.base_fusion_score for c in candidates_by_id.values()),
                default=0.0,
            )
            bonus_scale = peak_base if peak_base > 0.0 else 1.0
            for cid, g_score in graph_result.chunk_scores.items():
                ch = self.chunks_by_id.get(cid)
                if ch is None or not self._chunk_passes_filters(ch, cfg):
                    continue
                bonus = round(cfg.graph_weight * g_score * bonus_scale, 6)
                cand = candidates_by_id.get(cid)
                if cand is None:
                    candidates_by_id[cid] = ModularCandidate(
                        chunk_id=ch.chunk_id,
                        doc_source=ch.doc_source,
                        section_title=ch.section_title,
                        text=ch.text,
                        department=ch.department,
                        updated_year=ch.updated_year,
                        base_fusion_score=0.0,
                        graph_bonus=bonus,
                        combined_score=bonus,
                        rerank_score=None,
                        sources=["Knowledge-Graph"],
                    )
                else:
                    cand.graph_bonus = bonus
                    cand.combined_score = round(cand.base_fusion_score + bonus, 6)
                    if "Knowledge-Graph" not in cand.sources:
                        cand.sources.append("Knowledge-Graph")

        stage1_candidates = sorted(
            candidates_by_id.values(),
            key=lambda c: (c.combined_score, -c.chunk_id),
            reverse=True,
        )[: cfg.stage1_k]

        # Stage 3: Post-Retrieval Cross-Encoder Reranking
        if cfg.use_reranker and stage1_candidates:
            pseudo_s1 = [
                HybridSearchResult(
                    chunk_id=c.chunk_id,
                    doc_source=c.doc_source,
                    section_title=c.section_title,
                    text=c.text,
                    vector_score=0.0,
                    vector_rank=idx + 1,
                    bm25_score=0.0,
                    bm25_rank=idx + 1,
                    rrf_score=c.combined_score,
                    final_rank=idx + 1,
                )
                for idx, c in enumerate(stage1_candidates)
            ]
            reranked: list[RerankedCandidate] = self.reranker.rerank(
                query, pseudo_s1, top_k=len(pseudo_s1)
            )
            rerank_map = {r.chunk_id: r.rerank_score for r in reranked}
            final_candidates: list[ModularCandidate] = []
            for r in reranked[: cfg.final_top_k]:
                orig = candidates_by_id[r.chunk_id]
                orig.rerank_score = r.rerank_score
                final_candidates.append(orig)
            # Keep scores visible on stage1 items too
            for c in stage1_candidates:
                if c.chunk_id in rerank_map:
                    c.rerank_score = rerank_map[c.chunk_id]
        else:
            final_candidates = stage1_candidates[: cfg.final_top_k]

        context_lines = [
            f"[Chunk #{c.chunk_id} | {c.doc_source} > {c.section_title} | sources={','.join(c.sources)}]\n{c.text}"
            for c in final_candidates
        ]
        if graph_result.matched_triples:
            triple_summary = "; ".join(
                f"({t.subject} --[{t.predicate}]--> {t.obj})"
                for t, _ in graph_result.matched_triples[:6]
            )
            context_lines.append(f"[Knowledge Graph Facts]\n{triple_summary}")

        system_prompt = (
            "You are an expert RAG Tutor assistant.\n"
            "Answer the user's question concisely using ONLY the retrieved context and graph facts below.\n\n"
            + "\n\n".join(context_lines)
        )
        gen = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=query,
            model=model,
            enable_thinking=enable_thinking,
            retrieved_context=[c.text for c in final_candidates],
        )

        return ModularRAGTrace(
            query=query,
            config=cfg,
            expansion=expansion,
            graph_result=graph_result,
            sql_filter_preview=sql_preview,
            stage1_candidates=stage1_candidates,
            final_candidates=final_candidates,
            final_answer=gen.content,
            llm_model_used=gen.model,
            llm_backend_used=gen.backend,
            latency_ms=gen.latency_ms,
        )
