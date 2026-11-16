# Advanced Graph, Hybrid, and Modular RAG Architectures

## 1. Stage 1: Parallel Retrieval Techniques
Modern production **Modular RAG** systems go beyond a single dense vector index by querying multiple specialized retrieval engines in parallel:

- **Dense + Sparse Search**: Dense retrieval uses continuous vector embeddings (such as OpenAI `text-embedding-3` or `BAAI/bge-base-en-v1.5`) to capture broad semantic meaning, paraphrases, and synonyms. Sparse retrieval uses lexical algorithms such as **Okapi BM25** or learned sparse models like **SPLADE** to match exact words, acronyms, part numbers, and diagnostic codes (e.g., `ERR-VEC-768` or `CONVEX-ALPHA-70`).
- **Vector + Knowledge Graph Search (`GraphRAG`)**: Combines unstructured chunk similarity with structured **Knowledge Graph** traversal. While the vector engine pulls narrative text passages, the graph engine follows typed entity-relationship triples such as `(GraphRAG, traverses, Knowledge Graph)` or `(Cross-Encoder, refines, Top-50 Candidates)` across 1 to 2 hops to surface multi-entity facts that might be scattered across separate documents.
- **Vector + SQL / Metadata Filtering**: Uses structured database predicates (filtering by `department`, `author`, `updated_year`, or access-control tier) before or alongside vector similarity search. In **DuckDB**, a SQL `WHERE` clause narrows `rag_chunks` to authorized or recent records while executing `array_cosine_similarity` in a single query.

## 2. Stage 2: Score Fusion Techniques (RRF vs. Convex Combination)
Because lexical BM25 scores (unbounded positive floats like `4.85`), cosine similarities (bounded in `[-1.0, 1.0]`), and graph hop scores operate on incompatible mathematical scales, you must blend candidate lists carefully:

- **Reciprocal Rank Fusion (`RRF-K60`)**: The industry-standard scale-invariant merging method. Instead of relying on raw similarity numbers, RRF sums reciprocal ranks (`1 / (k + rank)`) across retrievers. A chunk that ranks near the top of both dense and sparse lists is promoted to the top regardless of score distribution skew.
- **Weighted Linear Combination (Convex Combination `CONVEX-ALPHA-70`)**: Normalizes each retriever's scores onto a shared `[0.0, 1.0]` interval (typically via Min-Max or Z-score normalization) and computes a convex sum controlled by weight `alpha`:
  `Convex_Score = alpha * norm_vector + (1 - alpha) * norm_bm25`
  For example, setting `alpha = 0.7` yields 70% Vector Score + 30% Keyword Score. Convex Combination preserves score magnitude gaps—meaning a massive exact keyword spike stays influential—when score distributions are well calibrated.

## 3. Stage 3: Pre- and Post-Retrieval Refinement
Modular RAG pipelines wrap parallel retrieval and score fusion with intelligent query shaping and precision filtering:

- **Query Expansion (`Multi-Query` and `HyDE`)**: Before running hybrid search, an LLM rewrites an ambiguous user prompt into 3 or 4 distinct query variations (**Multi-Query Expansion**) to catch different phrasings, or generates a plausible hypothetical answer passage (**HyDE — Hypothetical Document Embeddings**). Embedding the hypothetical answer aligns query vectors directly with answer-style document chunks instead of short question phrasing.
- **Cross-Encoder Reranking (`Top-50 -> Top-5`)**: After fusion merges the top candidate pool (for example, the top 15 to 50 chunks), a Cross-Encoder model (such as **Cohere Rerank** or **BGE-Reranker**) reads the user's question alongside every individual chunk using full token-level cross-attention, discarding false positives and keeping only the top 3 to 5 passages for the LLM.

## 4. Production Vector Databases & Tuning Guidelines
When scaling beyond embedded **DuckDB**, several vector databases support hybrid dense + sparse search and metadata filtering out of the box:

- **Qdrant**: Supports named dense and sparse vectors (`SPLADE` / `BM25`) with native payload filtering and built-in RRF fusion.
- **Weaviate**: Combines HNSW dense vectors with BM25F inverted indexes and supports both `relativeScoreFusion` (Convex) and `rankedFusion` (RRF).
- **Pinecone**: Provides hybrid dense-sparse indexes and managed rerankers via Pinecone Inference.
- **Elasticsearch / OpenSearch**: Pairs mature BM25 inverted indexes with dense `knn` vector search and RRF or linear score combiners.

To tune the keyword-versus-vector weight `alpha`, evaluate queries containing exact identifiers (`alpha = 0.2` to `0.4`), balanced conceptual and technical questions (`alpha = 0.5` to `0.7`), and broad paraphrased questions (`alpha = 0.75` to `0.9`), always pairing broad Stage-1 recall with a Stage-2 Cross-Encoder reranker.
