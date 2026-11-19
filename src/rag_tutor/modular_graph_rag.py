"""Advanced Graph, Hybrid, and Modular RAG Pipeline."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .chunker import Chunk
from .duckdb_store import DuckDBVectorStore
from .hybrid_search import HybridSearchEngine, HybridSearchResult
from .llm_backend import (
    LlamaMacOSClient,
    get_default_client,
    tokenize,
)
from .reranker import CrossEncoderReranker
from .scratch_rag import build_rag_prompt

ExpansionMode = Literal["none", "multi_query", "hyde"]
FusionMethod = Literal["rrf", "convex"]

_EXPLICIT_TRIPLE_RE = re.compile(
    r"\(([A-Za-z0-9][A-Za-z0-9 ./_-]{1,40}),\s*([a-zA-Z_][a-zA-Z0-9 _-]{1,32}),\s*([A-Za-z0-9][A-Za-z0-9 ./_>-]{1,48})\)"
)

_CANONICAL_TRIPLES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
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
)


@dataclass(frozen=True)
class KnowledgeTriple:
    """A directed `(subject, predicate, obj)` relationship grounded in a corpus chunk."""

    subject: str
    predicate: str
    obj: str
    source_chunk_id: int
    doc_source: str


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
    expanded_queries: list[str]
    hypothetical_doc: str = ""


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
    base_fusion_score: float
    combined_score: float
    rerank_score: float | None
    sources: list[str]


@dataclass
class ModularRAGTrace:
    """Execution trace across pre-retrieval, parallel retrieval, fusion, and refinement."""

    expansion: QueryExpansionResult
    graph_result: GraphSearchResult
    final_candidates: list[ModularCandidate]
    final_answer: str
    llm_model_used: str
    latency_ms: float


class KnowledgeGraphStore:
    """Lightweight Knowledge Graph built from corpus chunks for Vector + Graph RAG."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.triples: list[KnowledgeTriple] = []
        self._adj: dict[str, list[tuple[str, KnowledgeTriple]]] = defaultdict(list)
        self._entity_tokens: dict[str, set[str]] = {}
        self._build_from_chunks(chunks)

    @staticmethod
    def _find_best_chunk(
        chunk_texts: list[tuple[Chunk, str]], keywords: tuple[str, ...]
    ) -> Chunk | None:
        primary = keywords[0].lower()
        min_hits = min(2, len(keywords))
        kw_set = {k.lower() for k in keywords}
        best_chunk: Chunk | None = None
        best_hits = 0
        for ch, text_low in chunk_texts:
            if primary not in text_low:
                continue
            hits = sum(1 for kw in kw_set if kw in text_low)
            if hits >= min_hits and hits > best_hits:
                best_hits = hits
                best_chunk = ch
        return best_chunk

    def _add_triple(
        self, triple: KnowledgeTriple, seen: set[tuple[str, str, str]]
    ) -> None:
        key = (triple.subject.lower(), triple.predicate.lower(), triple.obj.lower())
        if key in seen:
            return
        seen.add(key)
        self.triples.append(triple)
        self._adj[triple.subject].append((triple.obj, triple))
        self._adj[triple.obj].append((triple.subject, triple))
        for ent in (triple.subject, triple.obj):
            if ent not in self._entity_tokens:
                self._entity_tokens[ent] = set(
                    tokenize(ent, stem=True, drop_stopwords=True)
                )

    def _build_from_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        chunk_texts = [(ch, f"{ch.section_title} {ch.text}".lower()) for ch in chunks]
        seen: set[tuple[str, str, str]] = set()
        for subj, pred, obj, kws in _CANONICAL_TRIPLES:
            if ch := self._find_best_chunk(chunk_texts, kws):
                self._add_triple(
                    KnowledgeTriple(subj, pred, obj, ch.chunk_id, ch.doc_source),
                    seen,
                )

        # Also extract explicit inline `(Subject, predicate, Object)` triples from text.
        for ch in chunks:
            for subj, pred, obj in _EXPLICIT_TRIPLE_RE.findall(ch.text):
                self._add_triple(
                    KnowledgeTriple(
                        subj.strip(),
                        pred.strip().replace(" ", "_"),
                        obj.strip(),
                        ch.chunk_id,
                        ch.doc_source,
                    ),
                    seen,
                )

    def search(
        self,
        query: str,
        max_hops: int = 2,
        max_triples: int = 12,
        allowed_chunk_ids: set[int] | None = None,
    ) -> GraphSearchResult:
        """Find entity seeds matching `query` and traverse up to `max_hops` graph edges."""
        if not query.strip():
            return GraphSearchResult([], [], {})
        max_hops = max(1, min(3, int(max_hops)))
        q_lower = query.lower()
        q_tokens = set(tokenize(query, stem=True, drop_stopwords=True))

        entity_scores: list[tuple[float, str]] = []
        for entity, e_tokens in self._entity_tokens.items():
            e_lower = entity.lower()
            score = (2.5 if len(e_lower) >= 3 and e_lower in q_lower else 0.0) + len(
                q_tokens & e_tokens
            )
            if score > 0.0:
                entity_scores.append((score, entity))

        entity_scores.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        seed_entities = [ent for _, ent in entity_scores[:4]]
        if not seed_entities:
            return GraphSearchResult([], [], {})

        visited_entities = {seed: 0 for seed in seed_entities}
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
                if (
                    allowed_chunk_ids is not None
                    and triple.source_chunk_id not in allowed_chunk_ids
                ):
                    continue
                t_key = (triple.subject, triple.predicate, triple.obj)
                if t_key not in seen_triples:
                    seen_triples.add(t_key)
                    matched_triples.append((triple, next_hop))
                    raw_chunk_scores[triple.source_chunk_id] += 0.5 ** (next_hop - 1)
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
        if not matched_triples:
            seed_entities = []
        return GraphSearchResult(seed_entities, matched_triples, norm_chunk_scores)


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
            return QueryExpansionResult("none", [cleaned] if cleaned else [])
        if mode == "multi_query":
            return self._expand_multi_query(cleaned, model=model)
        if mode == "hyde":
            return self._expand_hyde(cleaned, model=model)
        raise ValueError(f"Unsupported expansion mode {mode!r}")

    def _generate_if_online(
        self,
        system_prompt: str,
        query: str,
        model: str | None,
        max_tokens: int,
    ) -> str | None:
        if not model or model == "educational-fallback":
            return None
        res = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=query,
            model=model,
            max_tokens=max_tokens,
            retrieved_context=[],
        )
        return res.content.strip() if res.backend == "llama-macos" else None

    def _expand_multi_query(
        self, query: str, model: str | None = None
    ) -> QueryExpansionResult:
        content = self._generate_if_online(
            "Rewrite the user question into 3 concise search query variations covering "
            "different synonyms and exact technical terms. Return one query per line.",
            query,
            model,
            max_tokens=160,
        )
        variations: list[str] = []
        if content:
            for line in content.splitlines():
                item = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
                if len(item) >= 8 and item.lower() != query.lower():
                    variations.append(item)

        if not variations:
            key_terms = (
                " ".join(tokenize(query, stem=False, drop_stopwords=True)[:6]) or query
            )
            variations = [
                f"{key_terms} architecture workflow and pipeline stages",
                f"exact technical terms and configuration for {key_terms}",
                f"comparison and tuning best practices for {key_terms}",
            ]

        return QueryExpansionResult(
            mode="multi_query",
            expanded_queries=list(dict.fromkeys([query, *variations]))[:4],
        )

    def _expand_hyde(
        self, query: str, model: str | None = None
    ) -> QueryExpansionResult:
        content = self._generate_if_online(
            "Write a concise, factual 2-sentence technical passage that directly answers "
            "the user's question as if excerpted from an engineering reference manual.",
            query,
            model,
            max_tokens=180,
        )
        if not content or len(content) < 20:
            tokens = tokenize(query, stem=False, drop_stopwords=True)
            focus = ", ".join(tokens[:6]) if tokens else query
            content = (
                f"In production RAG architectures addressing {focus}, systems pair dense vector "
                "similarity search with lexical BM25 keyword scoring, optional Knowledge Graph "
                "traversal, and SQL metadata filtering. Candidate lists are merged via Reciprocal "
                "Rank Fusion (RRF-K60) or Min-Max normalized Convex Combination before a Stage-2 "
                "Cross-Encoder reranker refines the final top chunks for grounded LLM synthesis."
            )

        return QueryExpansionResult(
            mode="hyde", expanded_queries=[query], hypothetical_doc=content
        )


def _make_candidate(
    src: HybridSearchResult | Chunk,
    base_fusion_score: float,
    combined_score: float,
    sources: list[str],
) -> ModularCandidate:
    return ModularCandidate(
        chunk_id=src.chunk_id,
        doc_source=src.doc_source,
        section_title=src.section_title,
        text=src.text,
        base_fusion_score=base_fusion_score,
        combined_score=combined_score,
        rerank_score=None,
        sources=list(sources),
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
        self.embed_fn = embed_fn or store.embed_fn

    @staticmethod
    def _chunk_passes_filters(chunk: Chunk, config: ModularRAGConfig) -> bool:
        return (
            not config.department
            or config.department == "all"
            or chunk.department == config.department
        ) and (config.min_year is None or chunk.updated_year >= int(config.min_year))

    def run(
        self,
        query: str,
        config: ModularRAGConfig | None = None,
        model: str | None = None,
        enable_thinking: bool = False,
    ) -> ModularRAGTrace:
        """Execute the configured Modular RAG pipeline and return a full trace."""
        cfg = config or ModularRAGConfig()
        cleaned_query = query.strip()
        if not cleaned_query:
            return ModularRAGTrace(
                expansion=QueryExpansionResult("none", []),
                graph_result=GraphSearchResult([], [], {}),
                final_candidates=[],
                final_answer="Please enter a question to query the Modular RAG pipeline.",
                llm_model_used="educational-fallback",
                latency_ms=0.0,
            )

        alpha = max(0.0, min(1.0, float(cfg.alpha)))
        rrf_k = max(1, int(cfg.rrf_k))
        graph_weight = max(0.0, float(cfg.graph_weight))
        stage1_k = max(1, int(cfg.stage1_k))
        final_k = max(1, int(cfg.final_top_k))
        has_sql_filter = (
            bool(cfg.department and cfg.department != "all") or cfg.min_year is not None
        )

        expansion = self.expander.expand(
            cleaned_query, mode=cfg.expansion_mode, model=model
        )
        hyde_vec = (
            self.embed_fn(expansion.hypothetical_doc)
            if expansion.mode == "hyde" and expansion.hypothetical_doc
            else None
        )

        candidates_by_id: dict[int, ModularCandidate] = {}
        multi_scores: dict[int, float] = defaultdict(float)
        queries = expansion.expanded_queries or [cleaned_query]
        is_multi = len(queries) > 1

        if cfg.enable_dense_sparse:
            base_sources = [
                *(
                    ["HyDE-Dense" if hyde_vec is not None else "Dense"]
                    if alpha > 0.0
                    else []
                ),
                *(["BM25"] if alpha < 1.0 else []),
                *(["SQL-Filter"] if has_sql_filter else []),
                *(["Multi-Query"] if is_multi else []),
            ]
            for q_idx, q_variant in enumerate(queries):
                hits = self.hybrid_engine.search(
                    query=q_variant,
                    top_k=stage1_k,
                    alpha=alpha,
                    rrf_k=rrf_k,
                    fusion_method=cfg.fusion_method,
                    department=cfg.department,
                    min_year=cfg.min_year,
                    query_vec=hyde_vec if q_idx == 0 else None,
                )
                for rank_pos, h in enumerate(hits):
                    if is_multi:
                        multi_scores[h.chunk_id] += (
                            1.0 / (rrf_k + rank_pos + 1)
                            if cfg.fusion_method == "rrf"
                            else h.fusion_score / len(queries)
                        )
                    if h.chunk_id not in candidates_by_id:
                        candidates_by_id[h.chunk_id] = _make_candidate(
                            h, h.fusion_score, h.fusion_score, base_sources
                        )

            if is_multi:
                for cid, cand in candidates_by_id.items():
                    cand.base_fusion_score = round(multi_scores[cid], 6)
                    cand.combined_score = cand.base_fusion_score

        if cfg.enable_graph:
            allowed_ids = (
                {
                    cid
                    for cid, ch in self.chunks_by_id.items()
                    if self._chunk_passes_filters(ch, cfg)
                }
                if has_sql_filter
                else None
            )
            graph_result = self.graph_store.search(
                cleaned_query, max_hops=2, allowed_chunk_ids=allowed_ids
            )
        else:
            graph_result = GraphSearchResult([], [], {})

        if cfg.enable_graph and graph_result.chunk_scores:
            peak_base = max(
                (c.base_fusion_score for c in candidates_by_id.values()), default=0.0
            )
            bonus_scale = peak_base if peak_base > 0.0 else 1.0
            for cid, g_score in graph_result.chunk_scores.items():
                ch = self.chunks_by_id.get(cid)
                if ch is None:
                    continue
                bonus = round(graph_weight * g_score * bonus_scale, 6)
                cand = candidates_by_id.get(cid)
                if cand is None:
                    candidates_by_id[cid] = _make_candidate(
                        ch, 0.0, bonus, ["Knowledge-Graph"]
                    )
                else:
                    cand.combined_score = round(cand.base_fusion_score + bonus, 6)
                    if "Knowledge-Graph" not in cand.sources:
                        cand.sources.append("Knowledge-Graph")

        stage1_candidates = sorted(
            candidates_by_id.values(),
            key=lambda c: (c.combined_score, -c.chunk_id),
            reverse=True,
        )[:stage1_k]

        if cfg.use_reranker and stage1_candidates:
            for c in stage1_candidates:
                c.rerank_score, _ = self.reranker.score_pair(cleaned_query, c.text)
            final_candidates = sorted(
                stage1_candidates,
                key=lambda c: (c.rerank_score or 0.0, c.combined_score),
                reverse=True,
            )[:final_k]
        else:
            final_candidates = stage1_candidates[:final_k]

        context_blocks = [
            f"[Chunk #{c.chunk_id} | {c.doc_source} > {c.section_title} | sources={','.join(c.sources)}]\n{c.text}"
            for c in final_candidates
        ]
        if graph_result.matched_triples:
            triple_summary = "; ".join(
                f"({t.subject} --[{t.predicate}]--> {t.obj})"
                for t, _ in graph_result.matched_triples[:6]
            )
            context_blocks.append(f"[Knowledge Graph Facts]\n{triple_summary}")

        system_prompt, user_prompt = build_rag_prompt(cleaned_query, context_blocks)
        gen = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            enable_thinking=enable_thinking,
            retrieved_context=[c.text for c in final_candidates],
        )

        return ModularRAGTrace(
            expansion=expansion,
            graph_result=graph_result,
            final_candidates=final_candidates,
            final_answer=gen.content,
            llm_model_used=gen.model,
            latency_ms=gen.latency_ms,
        )
