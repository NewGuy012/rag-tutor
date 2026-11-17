"""RAG Tutor: Interactive educational toolkit for Retrieval-Augmented Generation."""

from .agentic_rag import AgenticRAGPipeline
from .chunker import chunk_document, load_sample_markdown_corpus
from .duckdb_store import DuckDBVectorStore
from .hybrid_search import HybridSearchEngine
from .llm_backend import (
    LlamaMacOSClient,
    LocalEmbeddingEngine,
    get_default_client,
    get_default_embedder,
)
from .modular_graph_rag import (
    KnowledgeGraphStore,
    ModularRAGConfig,
    ModularRAGPipeline,
    QueryExpander,
)
from .reranker import CrossEncoderReranker
from .scratch_rag import (
    SimpleVectorDB,
    build_rag_prompt,
    cosine_similarity,
    dot_product,
    load_ev_facts,
    vector_norm,
)
from .visualizations import (
    build_agentic_mermaid_diagram,
    build_modular_rag_mermaid,
    build_studio_pipeline_mermaid,
    render_2d_compass_svg,
    render_ab_arena_comparison_svg,
    render_chunk_timeline_svg,
    render_cross_attention_heatmap_svg,
    render_embedding_barcode_svg,
    render_embedding_space_2d_svg,
    render_faithfulness_audit_svg,
    render_fusion_comparison_svg,
    render_hybrid_rrf_svg,
    render_rerank_slopegraph_svg,
    render_subgraph_traversal_svg,
)

__all__ = [
    "AgenticRAGPipeline",
    "CrossEncoderReranker",
    "DuckDBVectorStore",
    "HybridSearchEngine",
    "KnowledgeGraphStore",
    "LlamaMacOSClient",
    "LocalEmbeddingEngine",
    "ModularRAGConfig",
    "ModularRAGPipeline",
    "QueryExpander",
    "SimpleVectorDB",
    "build_agentic_mermaid_diagram",
    "build_modular_rag_mermaid",
    "build_rag_prompt",
    "build_studio_pipeline_mermaid",
    "chunk_document",
    "cosine_similarity",
    "dot_product",
    "get_default_client",
    "get_default_embedder",
    "load_ev_facts",
    "load_sample_markdown_corpus",
    "render_2d_compass_svg",
    "render_ab_arena_comparison_svg",
    "render_chunk_timeline_svg",
    "render_cross_attention_heatmap_svg",
    "render_embedding_barcode_svg",
    "render_embedding_space_2d_svg",
    "render_faithfulness_audit_svg",
    "render_fusion_comparison_svg",
    "render_hybrid_rrf_svg",
    "render_rerank_slopegraph_svg",
    "render_subgraph_traversal_svg",
    "vector_norm",
]
