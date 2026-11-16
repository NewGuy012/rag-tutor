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
]
