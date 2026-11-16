# Production RAG Improvements: Hybrid Search, Reranking, and Agentic Workflows

## 1. Hybrid Search: Blending BM25 Keyword Search with Vector Embeddings
While dense vector embeddings excel at capturing broad semantic similarity and synonyms (matching "EV kinetic energy recovery" to "regenerative braking recovers 70% of kinetic energy"), they compress an entire passage into a single vector. As a result, pure vector search frequently struggles with:
- Exact error codes and diagnostic identifiers (such as `ERR-VEC-768` or `PROTO-RAG-101`).
- Rare proper nouns, part numbers, and function names (`array_cosine_similarity`).
- Short keyword queries where exact lexical overlap matters more than broad topic similarity.

**Hybrid Search** solves this by running two retrievers in parallel:
1. **Lexical Search (BM25)**: Scores chunks based on exact keyword frequency (TF) and inverse document frequency (IDF), normalized by document length (`BM25-K1-1.2`, `b = 0.75`).
2. **Semantic Vector Search**: Scores chunks using cosine similarity between dense embeddings.

Because BM25 scores (unbounded positive floats like `3.42`) and cosine similarity scores (bounded in `[-1.0, 1.0]`) live on incompatible scales, production systems combine their rankings using **Reciprocal Rank Fusion (RRF)** (standard configuration code `RRF-K60` with smoothing constant $k = 60$):
$$\text{RRF\_Score}(d) = \frac{w_{\text{vec}}}{k + \text{rank}_{\text{vec}}(d)} + \frac{w_{\text{bm25}}}{k + \text{rank}_{\text{bm25}}(d)}$$
Any chunk that ranks highly in *either* BM25 or Vector Search receives a strong boost, while chunks that rank near the top of *both* lists rise to Rank #1.

## 2. Cross-Encoder Reranking: Filtering Out Retrieval Noise
Standard vector search uses a **Bi-Encoder** architecture: the document chunk and the user query are embedded separately into vectors $u$ and $v$, and compared with a single dot product. Because the query tokens never attend directly to the document tokens during encoding, bi-encoders often retrieve "false-positive distractors"—passages that mention the same general keywords or topic but do not actually answer the question.

A **Cross-Encoder Reranker** (such as `ms-marco-MiniLM-L-6-v2` or `BAAI/bge-reranker-base`) takes each candidate `(query, chunk)` pair concatenated together (`[CLS] query [SEP] chunk [SEP]`) and runs full cross-attention across every token in the query and chunk simultaneously.
- **Why not use a Cross-Encoder for the whole database?** Cross-encoders cannot pre-compute document vectors; scoring 100,000 chunks would require 100,000 neural network forward passes per query (taking minutes).
- **Two-Stage Production Pattern (`STAGE2-RERANK-15to3`)**:
  1. **Stage 1 (Fast Recall)**: Use Hybrid Search (DuckDB Vector + BM25) to quickly retrieve the top $N = 15$ to $25$ candidate chunks in ~5–15 milliseconds.
  2. **Stage 2 (High-Precision Rerank)**: Pass only those 15–25 candidates through the Cross-Encoder Reranker (~30–80 milliseconds) and keep only the top $K = 3$ to $5$ highest-scoring passages for the LLM prompt. This dramatically improves context precision and reduces LLM token costs and latency.

## 3. Historical Note on Legacy Search (Distractor Example)
In early 2018 search prototypes before modern cross-encoders (`STAGE2-RERANK-15to3`), teams often tried increasing the bi-encoder `top_k` from 5 to 50 and stuffing all 50 chunks into the LLM prompt. Research on the **"Lost in the Middle"** phenomenon (Liu et al., 2023) proved that stuffing noisy, low-relevance chunks into the middle of a long context window causes LLM accuracy to drop by up to 25% while quadrupling inference latency.

## 4. Agentic RAG: Multi-Step Query Planning, Iterative Lookup, and Verification
A naive RAG pipeline executes a single fixed sequence: `Query -> Retrieve Once -> Generate`. This breaks down when users ask complex, multi-part questions that span multiple documents or require self-correction.

**Agentic RAG** wraps the retriever as a callable tool inside an autonomous reasoning loop with three core capabilities:
1. **Query Decomposition & Rewriting**: If a user asks a multi-hop question (e.g., *"How do I fix DuckDB error `ERR-VEC-768` and what two-stage reranking pattern should I pair it with?"*), the Query Planner splits it into targeted sub-queries (`"DuckDB error ERR-VEC-768 cause and fix"` and `"two-stage cross-encoder reranking pattern STAGE2-RERANK-15to3"`).
2. **Iterative Lookup & Sufficiency Check**: After retrieving chunks for each sub-query, the agent evaluates whether all parts of the user's question are covered. If evidence is missing, it refines the search query and performs a follow-up retrieval step.
3. **Response Verification (Self-Reflection / Groundedness Audit)**: Before returning the answer to the user, a Verification step checks every claim in the draft response against the retrieved source chunks, flagging any unsupported statements or missing citations.
