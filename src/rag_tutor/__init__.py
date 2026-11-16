"""RAG Tutor: Interactive educational toolkit for Retrieval-Augmented Generation."""

from .scratch_rag import (
    dot_product,
    vector_norm,
    cosine_similarity,
    SimpleVectorDB,
    load_ev_facts,
    build_rag_prompt,
)
from .llm_backend import (
    LlamaMacOSClient,
    LocalEmbeddingEngine,
    get_default_client,
    get_default_embedder,
)
from .chunker import (
    Chunk,
    chunk_by_lines,
    chunk_fixed_character,
    chunk_markdown_sections,
)
from .duckdb_store import DuckDBVectorStore
from .hybrid_search import HybridSearchEngine, HybridSearchResult
from .reranker import CrossEncoderReranker, RerankedCandidate
from .agentic_rag import AgenticRAGPipeline, AgenticTrace
from .visualizations import (
    build_agentic_mermaid_diagram,
    build_studio_pipeline_mermaid,
    render_2d_compass_svg,
    render_ab_arena_comparison_svg,
    render_chunk_timeline_svg,
    render_cross_attention_heatmap_svg,
    render_embedding_barcode_svg,
    render_embedding_space_2d_svg,
    render_faithfulness_audit_svg,
    render_hybrid_rrf_svg,
    render_rerank_slopegraph_svg,
)

__all__ = [
    "dot_product",
    "vector_norm",
    "cosine_similarity",
    "SimpleVectorDB",
    "load_ev_facts",
    "build_rag_prompt",
    "LlamaMacOSClient",
    "LocalEmbeddingEngine",
    "get_default_client",
    "get_default_embedder",
    "Chunk",
    "chunk_by_lines",
    "chunk_fixed_character",
    "chunk_markdown_sections",
    "DuckDBVectorStore",
    "HybridSearchEngine",
    "HybridSearchResult",
    "CrossEncoderReranker",
    "RerankedCandidate",
    "AgenticRAGPipeline",
    "AgenticTrace",
    "build_agentic_mermaid_diagram",
    "build_studio_pipeline_mermaid",
    "render_2d_compass_svg",
    "render_ab_arena_comparison_svg",
    "render_chunk_timeline_svg",
    "render_cross_attention_heatmap_svg",
    "render_embedding_barcode_svg",
    "render_embedding_space_2d_svg",
    "render_faithfulness_audit_svg",
    "render_hybrid_rrf_svg",
    "render_rerank_slopegraph_svg",
]
