# Vector & Full-Text Search with DuckDB

## 1. Why DuckDB for Local RAG?
When moving beyond a 50-line in-memory Python list (`VECTOR_DB = []`), many tutorials jump straight to heavy client-server vector databases or cloud services. **DuckDB** offers a zero-configuration, embedded SQL database that runs in-process inside Python (`import duckdb`) and stores your entire RAG corpus—text chunks, metadata JSON, vector embeddings, and full-text indexes—inside a single portable `.duckdb` file.

## 2. Storing Fixed-Size Embeddings in DuckDB
DuckDB natively supports fixed-size `FLOAT[N]` array columns. For example, if your embedding model produces 384-dimensional or 768-dimensional vectors, you define your table as:

```sql
CREATE TABLE rag_chunks (
    chunk_id INTEGER PRIMARY KEY,
    doc_source VARCHAR,
    section_title VARCHAR,
    chunk_text VARCHAR,
    embedding FLOAT[128]
);
```

To score chunks against a query vector, DuckDB provides the built-in `array_cosine_similarity(vec1, vec2)` function, which is vectorized and executes in milliseconds across tens of thousands of chunks without leaving SQL:

```sql
SELECT
    chunk_id,
    doc_source,
    chunk_text,
    array_cosine_similarity(embedding, $query_vec::FLOAT[128]) AS vector_score
FROM rag_chunks
ORDER BY vector_score DESC
LIMIT 5;
```

## 3. Troubleshooting Common DuckDB Vector Errors
When building a DuckDB vector pipeline, watch out for two specific diagnostic codes:
- **Diagnostic `ERR-VEC-768` (Dimension Mismatch)**: Occurs when you index documents using a 768-dimension embedding model (like `bge-base-en-v1.5`) but query the table with a 384-dimension model (like `all-MiniLM-L6-v2`). Always store the embedding model identifier in your table metadata and enforce a single embedding dimension across indexing and retrieval.
- **Diagnostic `ERR-CHUNK-004` (Empty or Null Vector Norm)**: Occurs when whitespace-only chunks are embedded and produce a zero vector (`[0.0, 0.0, ...]`), causing `array_cosine_similarity` to return `NULL` or `NaN` because of division by zero. Always strip and filter chunks shorter than 15 characters before indexing.

## 4. DuckDB Full-Text Search (BM25) and the `vss` Extension
In addition to exact `array_cosine_similarity` scans, DuckDB provides two powerful extensions for scaling RAG:
1. **`fts` (Full-Text Search)**: Builds an inverted index over `chunk_text` and calculates Okapi **BM25** lexical relevance scores using `fts_main_rag_chunks.match_bm25(chunk_id, 'query terms')`. Default BM25 parameters in DuckDB are `k1 = 1.2` (term frequency saturation, code `BM25-K1-1.2`) and `b = 0.75` (document length normalization).
2. **`vss` (Vector Similarity Search)**: Adds Hierarchical Navigable Small World (**HNSW**) approximate nearest-neighbor indexes on `FLOAT[N]` columns for sub-millisecond retrieval on million-row tables.
