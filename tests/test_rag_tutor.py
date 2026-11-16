"""Minimal unit test suite for rag_tutor."""

from __future__ import annotations

import unittest

from rag_tutor import (
    AgenticRAGPipeline,
    CrossEncoderReranker,
    DuckDBVectorStore,
    HybridSearchEngine,
    LlamaMacOSClient,
    LocalEmbeddingEngine,
    SimpleVectorDB,
    build_agentic_mermaid_diagram,
    build_rag_prompt,
    chunk_document,
    cosine_similarity,
    dot_product,
    load_ev_facts,
    load_sample_markdown_corpus,
    render_embedding_barcode_svg,
    render_hybrid_rrf_svg,
    render_rerank_slopegraph_svg,
    vector_norm,
)


class TestRAGTutor(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.embedder = LocalEmbeddingEngine(dim=128)
        cls.chunks = load_sample_markdown_corpus(chunk_size=480, chunk_overlap=70)
        cls.store = DuckDBVectorStore(
            db_path=":memory:",
            embedding_dim=cls.embedder.dim,
            embed_fn=cls.embedder.embed,
        )
        cls.store.index_chunks(cls.chunks)
        cls.hybrid = HybridSearchEngine(cls.store)
        cls.reranker = CrossEncoderReranker()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.store.close()

    def test_vector_math_and_scratch_db(self) -> None:
        self.assertAlmostEqual(dot_product([1.0, 2.0], [3.0, 4.0]), 11.0)
        self.assertAlmostEqual(vector_norm([3.0, 4.0]), 5.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)
        with self.assertRaises(ValueError):
            dot_product([1.0], [1.0, 2.0])

        facts = load_ev_facts()
        db = SimpleVectorDB()
        self.assertEqual(db.index_chunks(facts[:4], self.embedder.embed), 4)
        hits = db.retrieve(self.embedder.embed("regenerative braking"), top_n=2)
        self.assertEqual(len(hits), 2)
        sys_p, usr_p = build_rag_prompt("q", hits)
        self.assertIn("Fact #1", sys_p)
        self.assertEqual(usr_p, "q")

    def test_chunking_dispatch(self) -> None:
        md = "# Title\n\nParagraph one with enough characters.\n\n## Section 2\n\nMore details here."
        md_chunks = chunk_document(md, strategy="markdown", chunk_size=100, chunk_overlap=20)
        fx_chunks = chunk_document(md, strategy="fixed", chunk_size=60, chunk_overlap=15)
        self.assertGreaterEqual(len(md_chunks), 2)
        self.assertGreaterEqual(len(fx_chunks), 1)

    def test_duckdb_search_and_sql_security(self) -> None:
        vec_hits = self.store.search_vector("array_cosine_similarity in DuckDB", top_k=3)
        bm25_hits = self.store.search_bm25("ERR-VEC-768", top_k=3)
        self.assertEqual(len(vec_hits), 3)
        self.assertGreater(bm25_hits[0].score, 0.0)

        rows = self.store.run_custom_sql("SELECT COUNT(*) AS n FROM rag_chunks;")
        self.assertEqual(rows[0]["n"], len(self.chunks))

        for unsafe_sql in (
            "DROP TABLE rag_chunks",
            "DELETE FROM rag_chunks",
            "SELECT 1; DROP TABLE rag_chunks",
            "SELECT * FROM read_csv('pyproject.toml')",
            "COPY rag_chunks TO '/tmp/out.csv'",
        ):
            with self.assertRaises(ValueError):
                self.store.run_custom_sql(unsafe_sql)

    def test_hybrid_rerank_agentic_and_visuals(self) -> None:
        h_res = self.hybrid.search("What is ERR-VEC-768?", top_k=5, alpha=0.5)
        self.assertEqual(len(h_res), 5)
        reranked = self.reranker.rerank("What is ERR-VEC-768?", h_res, top_k=3)
        self.assertEqual(len(reranked), 3)

        self.assertIn("<svg", render_hybrid_rrf_svg(h_res, 0.5, 60))
        self.assertIn("<svg", render_rerank_slopegraph_svg(reranked, 2))
        v = self.embedder.embed("ERR-VEC-768")
        self.assertIn("<svg", render_embedding_barcode_svg(v, v, 1.0))

        client = LlamaMacOSClient()
        with self.assertRaises(ValueError):
            LlamaMacOSClient(base_url="file:///etc/passwd")

        pipeline = AgenticRAGPipeline(self.hybrid, self.reranker, client)
        trace = pipeline.run(
            "What causes ERR-VEC-768 and how does STAGE2-RERANK-15to3 work?",
            model="educational-fallback",
        )
        self.assertTrue(trace.is_multi_hop)
        self.assertGreaterEqual(len(trace.planned_sub_queries), 2)
        self.assertIn("flowchart LR", build_agentic_mermaid_diagram(trace))


if __name__ == "__main__":
    unittest.main()
