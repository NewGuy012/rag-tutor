"""Minimal unit test suite for rag_tutor."""

from __future__ import annotations

import unittest

from rag_tutor import (
    AgenticRAGPipeline,
    CrossEncoderReranker,
    DuckDBVectorStore,
    HybridSearchEngine,
    KnowledgeGraphStore,
    LlamaMacOSClient,
    LocalEmbeddingEngine,
    ModularRAGConfig,
    ModularRAGPipeline,
    QueryExpander,
    SimpleVectorDB,
    build_agentic_mermaid_diagram,
    build_modular_rag_mermaid,
    build_rag_prompt,
    chunk_document,
    cosine_similarity,
    dot_product,
    load_ev_facts,
    load_sample_markdown_corpus,
    render_embedding_barcode_svg,
    render_fusion_comparison_svg,
    render_hybrid_rrf_svg,
    render_rerank_slopegraph_svg,
    render_subgraph_traversal_svg,
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

    def test_metadata_filtering_and_convex_fusion(self) -> None:
        meta = self.store.list_metadata_values()
        self.assertIn("advanced-architecture", meta["departments"])
        filtered = self.store.search_vector_filtered(
            "Convex Combination and HyDE",
            top_k=4,
            department="advanced-architecture",
            min_year=2026,
        )
        self.assertGreaterEqual(len(filtered), 1)
        self.assertTrue(all(r.department == "advanced-architecture" for r in filtered))
        self.assertTrue(all(r.updated_year >= 2026 for r in filtered))

        rrf_hits = self.hybrid.search("CONVEX-ALPHA-70", top_k=5, alpha=0.7, fusion_method="rrf")
        conv_hits = self.hybrid.search("CONVEX-ALPHA-70", top_k=5, alpha=0.7, fusion_method="convex")
        self.assertEqual(len(rrf_hits), 5)
        self.assertEqual(len(conv_hits), 5)
        self.assertTrue(all(0.0 <= h.norm_vector_score <= 1.0 for h in conv_hits))
        self.assertTrue(all(0.0 <= h.norm_bm25_score <= 1.0 for h in conv_hits))
        self.assertEqual(conv_hits[0].fusion_method, "convex")
        self.assertIn("<svg", render_fusion_comparison_svg(rrf_hits, conv_hits, 0.7, 60))

        with self.assertRaises(ValueError):
            self.hybrid.search("test", fusion_method="invalid")

    def test_graph_expansion_and_modular_rag(self) -> None:
        graph = KnowledgeGraphStore(self.chunks)
        self.assertGreaterEqual(len(graph.triples), 10)
        g_res = graph.search("How does DuckDB relate to RRF-K60 and Cross-Encoder?", max_hops=2)
        self.assertGreaterEqual(len(g_res.seed_entities), 1)
        self.assertGreaterEqual(len(g_res.matched_triples), 1)
        self.assertIn("<svg", render_subgraph_traversal_svg(g_res))

        client = LlamaMacOSClient()
        expander = QueryExpander(client)
        mq = expander.expand("How does GraphRAG work?", mode="multi_query", model="educational-fallback")
        self.assertGreaterEqual(len(mq.expanded_queries), 3)
        hyde = expander.expand("How does HyDE improve retrieval?", mode="hyde", model="educational-fallback")
        self.assertTrue(len(hyde.hypothetical_doc) > 30)

        modular = ModularRAGPipeline(
            store=self.store,
            chunks=self.chunks,
            hybrid_engine=self.hybrid,
            graph_store=graph,
            reranker=self.reranker,
            expander=expander,
            llm_client=client,
            embed_fn=self.embedder.embed,
        )
        cfg = ModularRAGConfig(
            expansion_mode="hyde",
            enable_dense_sparse=True,
            enable_graph=True,
            fusion_method="convex",
            alpha=0.7,
            use_reranker=True,
            final_top_k=3,
        )
        trace = modular.run(
            "How does HyDE work with Convex Combination and Cross-Encoder reranking?",
            config=cfg,
            model="educational-fallback",
        )
        self.assertEqual(len(trace.final_candidates), 3)
        self.assertTrue(all(c.rerank_score is not None for c in trace.final_candidates))
        self.assertIn("flowchart LR", build_modular_rag_mermaid(cfg, trace.expansion, trace.graph_result))


if __name__ == "__main__":
    unittest.main()
