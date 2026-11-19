import marimo

__generated_with = "0.10.0"
app = marimo.App(
    width="medium", app_title="RAG Tutor 05: Advanced Graph, Hybrid & Modular RAG"
)


@app.cell
def _():
    import html
    import sys
    from pathlib import Path

    import marimo as mo

    _repo_root = Path(__file__).resolve().parents[1]
    _src_dir = _repo_root / "src"
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    from rag_tutor import (
        DuckDBVectorStore,
        HybridSearchEngine,
        KnowledgeGraphStore,
        ModularRAGConfig,
        ModularRAGPipeline,
        build_modular_rag_mermaid,
        get_default_client,
        load_sample_markdown_corpus,
        render_fusion_comparison_svg,
        render_subgraph_traversal_svg,
    )

    llm_client = get_default_client()
    corpus_chunks = load_sample_markdown_corpus(
        strategy="markdown", chunk_size=480, chunk_overlap=70
    )
    store = DuckDBVectorStore()
    store.index_chunks(corpus_chunks)
    hybrid_engine = HybridSearchEngine(store)
    graph_store = KnowledgeGraphStore(corpus_chunks)
    modular_pipeline = ModularRAGPipeline(
        store=store,
        chunks=corpus_chunks,
        hybrid_engine=hybrid_engine,
        graph_store=graph_store,
        llm_client=llm_client,
    )
    metadata_vals = store.list_metadata_values()
    return (
        ModularRAGConfig,
        build_modular_rag_mermaid,
        corpus_chunks,
        graph_store,
        html,
        hybrid_engine,
        llm_client,
        metadata_vals,
        mo,
        modular_pipeline,
        render_fusion_comparison_svg,
        render_subgraph_traversal_svg,
        store,
    )


@app.cell
def _(llm_client, mo):
    is_online, _models, status_msg = llm_client.check_health()
    _opts = _models or ["educational-fallback"]
    model_dropdown = mo.ui.dropdown(
        options=_opts,
        value=_opts[0],
        label="Llama.app Model (`/v1/models`):",
    )
    thinking_toggle = mo.ui.switch(value=False, label="Enable Thinking")
    return is_online, model_dropdown, status_msg, thinking_toggle


@app.cell
def _(
    corpus_chunks,
    graph_store,
    html,
    is_online,
    mo,
    model_dropdown,
    status_msg,
    thinking_toggle,
):
    _badge = (
        "🟢 **Llama.app Connected**"
        if is_online
        else "🟡 **Offline Educational Fallback Active**"
    )
    mo.vstack(
        [
            mo.md(
                """
                # 🕸️ RAG Tutor — Notebook 5: Advanced Graph, Hybrid & Modular RAG
                ### *Parallel Retrieval, Score Fusion (RRF vs. Convex), Query Expansion (`HyDE` & `Multi-Query`) & Knowledge Graphs*

                Modern production retrieval goes far beyond a single vector lookup. **Modular RAG** combines
                multiple specialized retrieval engines in parallel, normalizes and merges their rankings, and
                wraps retrieval in pre- and post-retrieval refinement stages:

                1. **Stage 1 — Parallel Retrieval**: **Dense + Sparse** (`BM25` / `SPLADE`), **Vector + Knowledge Graph (`GraphRAG`)**, and **Vector + SQL/Metadata Filtering**.
                2. **Stage 2 — Score Fusion**: **Reciprocal Rank Fusion (`RRF-K60`)** vs. **Weighted Linear (Convex) Combination** (`α * norm_vector + (1 - α) * norm_bm25`).
                3. **Stage 3 — Pre- & Post-Retrieval Refinement**: **Query Expansion (`HyDE` & `Multi-Query`)** and **Cross-Encoder Reranking** (`Top-50 -> Top-5`).
                """
            ),
            mo.md(
                f"---\n**Corpus Loaded**: `{len(corpus_chunks)}` chunks indexed in DuckDB · `{len(graph_store.triples)}` Knowledge Graph triples extracted\n\n"
                f"**Local LLM Backend**: {_badge} — _{html.escape(status_msg)}_"
            ),
            mo.hstack([model_dropdown, thinking_toggle], justify="start", gap=2),
        ]
    )


@app.cell
def _(metadata_vals, mo):
    parallel_query_input = mo.ui.text(
        value="How does DuckDB combine vector search with RRF-K60, GraphRAG triples, and Cross-Encoder reranking?",
        label="Stage 1 Parallel Retrieval Query:",
        full_width=True,
    )
    dept_options = {
        "All Departments": "all",
        **{d: d for d in metadata_vals["departments"]},
    }
    dept_filter = mo.ui.dropdown(
        options=dept_options,
        value="All Departments",
        label="SQL Metadata Filter (`department`):",
    )
    min_year_filter = mo.ui.slider(
        start=2024,
        stop=2026,
        step=1,
        value=2024,
        label="SQL Filter (`updated_year >=`):",
    )
    graph_hops_ctl = mo.ui.slider(
        start=1,
        stop=2,
        step=1,
        value=2,
        label="Knowledge Graph Max Hops:",
    )
    return (
        dept_filter,
        dept_options,
        graph_hops_ctl,
        min_year_filter,
        parallel_query_input,
    )


@app.cell
def _(
    dept_filter,
    graph_hops_ctl,
    graph_store,
    html,
    min_year_filter,
    mo,
    parallel_query_input,
    render_subgraph_traversal_svg,
    store,
):
    def _rank_rows(hits, fmt: str) -> str:
        return (
            "\n".join(
                f"| **#{i + 1}** | Chunk #{r.chunk_id} | `{r.score:{fmt}}` | `{html.escape(r.doc_source)}` > **{html.escape(r.section_title).replace('|', '\\|')}** |"
                for i, r in enumerate(hits)
            )
            or "| — | — | — | _Enter a query to search_ |"
        )

    _q = parallel_query_input.value
    _dense_hits = store.search_vector(_q, top_k=5)
    _sparse_hits = store.search_bm25(_q, top_k=5)
    _graph_res = graph_store.search(_q, max_hops=graph_hops_ctl.value)
    _filtered_hits = store.search_vector(
        _q,
        top_k=5,
        department=dept_filter.value,
        min_year=min_year_filter.value,
    )
    _sql_str, _sql_params = store.build_filtered_vector_sql(
        department=dept_filter.value,
        min_year=min_year_filter.value,
    )

    _dense_rows = _rank_rows(_dense_hits, ".4f")
    _sparse_rows = _rank_rows(_sparse_hits, ".3f")
    _graph_svg = render_subgraph_traversal_svg(_graph_res)
    _triple_rows = (
        "\n".join(
            f"| **{hop}-hop** | `{html.escape(t.subject).replace('|', '\\|')}` | `--[{html.escape(t.predicate).replace('|', '\\|')}]-->` | `{html.escape(t.obj).replace('|', '\\|')}` | Chunk #{t.source_chunk_id} (`{html.escape(t.doc_source)}`) |"
            for t, hop in _graph_res.matched_triples
        )
        or "| — | _No triples matched_ | — | — | — |"
    )

    _sql_rows = (
        "\n".join(
            f"| **#{i + 1}** | Chunk #{r.chunk_id} | `{html.escape(r.department)}` | `{r.updated_year}` | `{r.score:.4f}` | `{html.escape(r.doc_source)}` > {html.escape(r.section_title).replace('|', '\\|')} |"
            for i, r in enumerate(_filtered_hits)
        )
        or "| — | — | _No rows matched SQL filter_ | — | — | — |"
    )

    _tabs = mo.ui.tabs(
        {
            "1️⃣ Dense + Sparse Search": mo.md(
                "**Dense Vector Search** (`DuckDB FLOAT[128]`) captures conceptual meaning and synonyms, "
                "while **Sparse Search** (`Okapi BM25`) catches exact error codes, identifiers, and acronyms.\n\n"
                "| Dense Rank | Chunk | Cosine Score | Source Section |\n"
                "| :--- | :--- | :--- | :--- |\n"
                f"{_dense_rows}\n\n"
                "| Sparse Rank | Chunk | BM25 Score | Source Section |\n"
                "| :--- | :--- | :--- | :--- |\n"
                f"{_sparse_rows}"
            ),
            "2️⃣ Vector + Knowledge Graph (GraphRAG)": mo.vstack(
                [
                    mo.md(
                        f"**Seed Entities Matched**: {', '.join(f'`{html.escape(s)}`' for s in _graph_res.seed_entities) or '_None_'}"
                    ),
                    mo.Html(_graph_svg),
                    mo.md(
                        "| Hop | Subject | Predicate | Object | Grounded Source Chunk |\n"
                        "| :--- | :--- | :--- | :--- | :--- |\n" + _triple_rows
                    ),
                ]
            ),
            "3️⃣ Vector + SQL / Metadata Filtering": mo.vstack(
                [
                    mo.md(
                        "Narrow the candidate space with structured DuckDB SQL predicates **alongside** `array_cosine_similarity`:"
                    ),
                    mo.Html(
                        '<pre style="background:#020617;color:#e2e8f0;padding:10px 12px;border-radius:8px;'
                        'border:1px solid #1e293b;white-space:pre-wrap;font-size:12px;">'
                        f"{html.escape(_sql_str)}\n-- Bound parameters: {html.escape(repr(_sql_params))}</pre>"
                    ),
                    mo.md(
                        "| Rank | Chunk | `department` | `updated_year` | Cosine Score | Source Section |\n"
                        "| :--- | :--- | :--- | :--- | :--- | :--- |\n" + _sql_rows
                    ),
                ]
            ),
        }
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 1. Stage 1: Parallel Retrieval Techniques

                At the search stage, a Hybrid / Modular RAG system queries multiple specialized engines simultaneously.
                Adjust the query, SQL metadata filters, and graph traversal depth below, then inspect each tab:
                """
            ),
            parallel_query_input,
            mo.hstack(
                [dept_filter, min_year_filter, graph_hops_ctl],
                justify="start",
                gap=2,
            ),
            _tabs,
        ]
    )


@app.cell
def _(mo):
    fusion_query_preset = mo.ui.dropdown(
        options={
            "Exact Identifier + Concept: CONVEX-ALPHA-70 and linear score blending": "How does CONVEX-ALPHA-70 normalize and blend dense vector and BM25 scores?",
            "Diagnostic Code: ERR-VEC-768 and DuckDB array_cosine_similarity": "What triggers ERR-VEC-768 in DuckDB array_cosine_similarity?",
            "Scale-Free Merging: RRF-K60 vs Convex Combination": "Why use Reciprocal Rank Fusion RRF-K60 instead of raw score addition?",
        },
        value="Exact Identifier + Concept: CONVEX-ALPHA-70 and linear score blending",
        label="Fusion Comparison Query:",
    )
    fusion_alpha_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.1,
        value=0.7,
        label="Weight α (0.7 = 70% Vector + 30% Keyword):",
    )
    fusion_rrf_k_slider = mo.ui.slider(
        start=10,
        stop=100,
        step=10,
        value=60,
        label="RRF Smoothing Constant k:",
    )
    return fusion_alpha_slider, fusion_query_preset, fusion_rrf_k_slider


@app.cell
def _(
    fusion_alpha_slider,
    fusion_query_preset,
    fusion_rrf_k_slider,
    html,
    hybrid_engine,
    mo,
    render_fusion_comparison_svg,
):
    _q = fusion_query_preset.value
    _alpha = fusion_alpha_slider.value
    _k = fusion_rrf_k_slider.value

    _rrf_res = hybrid_engine.search(
        _q, top_k=6, alpha=_alpha, rrf_k=_k, fusion_method="rrf"
    )
    _conv_res = hybrid_engine.search(
        _q, top_k=6, alpha=_alpha, rrf_k=_k, fusion_method="convex"
    )
    _fusion_svg = render_fusion_comparison_svg(
        _rrf_res, _conv_res, alpha=_alpha, rrf_k=_k
    )

    _conv_rows = "\n".join(
        f"| **#{r.final_rank}** | Chunk #{r.chunk_id} | `{r.vector_score:.3f}` → **`{r.norm_vector_score:.3f}`** | `{r.bm25_score:.2f}` → **`{r.norm_bm25_score:.3f}`** | **`{r.convex_score:.4f}`** | `{r.rrf_score:.5f}` | `{html.escape(r.doc_source)}` > {html.escape(r.section_title).replace('|', '\\|')} |"
        for r in _conv_res
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 2. Stage 2: Score Fusion Arena — `RRF` vs. `Convex Combination`

                Because cosine similarity (`[-1, 1]`) and Okapi BM25 (`[0, ∞)`) sit on incompatible scales:
                - **Reciprocal Rank Fusion (`RRF`)** ignores raw scores and blends **rank positions**: $\\frac{\\alpha}{k + \\text{rank}_{\\text{vec}}} + \\frac{1 - \\alpha}{k + \\text{rank}_{\\text{bm25}}}$
                - **Weighted Linear (Convex) Combination** Min-Max normalizes both signals to $[0, 1]$ first, then computes $\\alpha \\cdot \\hat{S}_{\\text{vec}} + (1 - \\alpha) \\cdot \\hat{S}_{\\text{bm25}}$ (e.g., **70% Vector + 30% Keyword** at $\\alpha = 0.7$).
                """
            ),
            fusion_query_preset,
            mo.hstack(
                [fusion_alpha_slider, fusion_rrf_k_slider], justify="start", gap=2
            ),
            mo.Html(_fusion_svg),
            mo.md(
                "| Convex Rank | Chunk | Raw → Norm Vector | Raw → Norm BM25 | Convex Score | RRF Score | Source Section |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n" + _conv_rows
            ),
        ]
    )


@app.cell
def _(dept_options, mo):
    modular_query_input = mo.ui.text(
        value="What are the best practices for tuning keyword vs vector weights and adding a reranker with HyDE or GraphRAG?",
        label="End-to-End Modular RAG Query:",
        full_width=True,
    )
    expansion_mode_ctl = mo.ui.dropdown(
        options={
            "None (Direct Query)": "none",
            "Multi-Query Expansion (3–4 Variations)": "multi_query",
            "HyDE (Hypothetical Document Embedding)": "hyde",
        },
        value="HyDE (Hypothetical Document Embedding)",
        label="Pre-Retrieval Query Expansion:",
    )
    modular_fusion_ctl = mo.ui.dropdown(
        options={
            "Reciprocal Rank Fusion (RRF-K60)": "rrf",
            "Weighted Linear / Convex Combination": "convex",
        },
        value="Weighted Linear / Convex Combination",
        label="Score Fusion Strategy:",
    )
    modular_alpha_ctl = mo.ui.slider(
        0.0, 1.0, step=0.1, value=0.7, label="Dense ↔ Sparse α:"
    )
    modular_enable_graph_ctl = mo.ui.switch(
        value=True, label="Enable Knowledge Graph Traversal"
    )
    modular_rerank_ctl = mo.ui.switch(
        value=True, label="Enable Stage-2 Cross-Encoder Reranking"
    )
    modular_dept_ctl = mo.ui.dropdown(
        options=dept_options,
        value="All Departments",
        label="SQL Metadata Department:",
    )
    return (
        expansion_mode_ctl,
        modular_alpha_ctl,
        modular_dept_ctl,
        modular_enable_graph_ctl,
        modular_fusion_ctl,
        modular_query_input,
        modular_rerank_ctl,
    )


@app.cell
def _(
    ModularRAGConfig,
    build_modular_rag_mermaid,
    expansion_mode_ctl,
    html,
    mo,
    model_dropdown,
    modular_alpha_ctl,
    modular_dept_ctl,
    modular_enable_graph_ctl,
    modular_fusion_ctl,
    modular_pipeline,
    modular_query_input,
    modular_rerank_ctl,
    thinking_toggle,
):
    _cfg = ModularRAGConfig(
        expansion_mode=expansion_mode_ctl.value,
        enable_dense_sparse=True,
        enable_graph=modular_enable_graph_ctl.value,
        department=modular_dept_ctl.value,
        fusion_method=modular_fusion_ctl.value,
        alpha=modular_alpha_ctl.value,
        use_reranker=modular_rerank_ctl.value,
        stage1_k=10,
        final_top_k=4,
    )
    _trace = modular_pipeline.run(
        query=modular_query_input.value,
        config=_cfg,
        model=model_dropdown.value,
        enable_thinking=thinking_toggle.value,
    )

    _mermaid = mo.mermaid(
        build_modular_rag_mermaid(
            config=_cfg,
            expansion=_trace.expansion,
            graph_result=_trace.graph_result,
        )
    )

    if _trace.expansion.mode == "multi_query":
        _exp_block = mo.callout(
            mo.md(
                "**Multi-Query Variations Executed**:\n"
                + "\n".join(
                    f"{i + 1}. `{html.escape(q)}`"
                    for i, q in enumerate(_trace.expansion.expanded_queries)
                )
            ),
            kind="info",
        )
    elif _trace.expansion.mode == "hyde":
        _exp_block = mo.callout(
            mo.vstack(
                [
                    mo.md("**HyDE Hypothetical Document Embedded for Dense Search**:"),
                    mo.Html(
                        f'<div style="white-space:pre-wrap;">{html.escape(_trace.expansion.hypothetical_doc)}</div>'
                    ),
                ]
            ),
            kind="info",
        )
    else:
        _exp_block = mo.md("_Direct query mode active (no query expansion applied)._")

    _cand_rows = (
        "\n".join(
            f"| **#{i + 1}** | Chunk #{c.chunk_id} | {', '.join(f'`{html.escape(s)}`' for s in c.sources)} | `{c.combined_score:.4f}` | {f'`{c.rerank_score:.4f}`' if c.rerank_score is not None else '_Bypassed_'} | `{html.escape(c.doc_source)}` > **{html.escape(c.section_title).replace('|', '\\|')}** |"
            for i, c in enumerate(_trace.final_candidates)
        )
        or "| — | — | — | — | — | _No candidates retrieved_ |"
    )

    mo.vstack(
        [
            mo.md(
                """
                ## 3. Stage 3 & Full Modular RAG Composer (`HyDE`, `Multi-Query` & Cross-Encoder Reranking)

                Toggle **Query Expansion**, **Knowledge Graph**, **Score Fusion**, **SQL Metadata Filtering**, and **Cross-Encoder Reranking**
                below to watch the **Modular RAG Architecture** rewire and inspect the final grounded answer:
                """
            ),
            modular_query_input,
            mo.hstack(
                [expansion_mode_ctl, modular_fusion_ctl, modular_dept_ctl],
                justify="start",
                gap=2,
            ),
            mo.hstack(
                [modular_alpha_ctl, modular_enable_graph_ctl, modular_rerank_ctl],
                justify="start",
                gap=3,
            ),
            mo.md("### Live Modular Pipeline Architecture (`mo.mermaid`)"),
            _mermaid,
            mo.md("### Pre-Retrieval Expansion Output"),
            _exp_block,
            mo.md(
                f"### Final Ranked Context Sent to LLM (`{len(_trace.final_candidates)}` Chunks)\n"
                "| Rank | Chunk | Active Retrieval Provenance | Stage-1/2 Fused Score | Stage-2 Rerank Score | Document & Section |\n"
                "| :--- | :--- | :--- | :--- | :--- | :--- |\n" + _cand_rows
            ),
            mo.md(
                f"### Grounded Answer (`{html.escape(_trace.llm_model_used)}` · `{_trace.latency_ms:.0f} ms`)"
            ),
            mo.callout(
                mo.Html(
                    f'<div style="white-space:pre-wrap;">{html.escape(_trace.final_answer)}</div>'
                ),
                kind="success",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Next-Step Production Guide: Vector Databases, Weight Tuning & Rerankers

    | Dimension | Embedded DuckDB (Local Curriculum) | Qdrant | Weaviate | Pinecone | Elasticsearch / OpenSearch |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Hybrid Support** | `FLOAT[N]` + `fts` BM25 + SQL | Native Dense + Sparse (`SPLADE`) | Native HNSW + BM25F | Sparse-Dense Hybrid Index | Dense `knn` + Lucene BM25 |
    | **Built-In Fusion** | Custom SQL / Python (`RRF` or `Convex`) | Native `RRF` (`Prefetch`) | `rankedFusion` & `relativeScoreFusion` | Client/Alpha weighting | `rrf` & `linear` retrievers |
    | **Metadata Filter** | Full SQL `WHERE` joins & predicates | Payload indexes (pre-filter in HNSW) | GraphQL / Filterable HNSW | Metadata filter dicts | Native inverted filter cache |
    | **Best Used When** | Local/desktop RAG, analytics, zero ops | High-throughput open-source hybrid search | Schema-driven multi-modal hybrid apps | Fully managed serverless scale | Enterprise log/text search + vectors |

    ### Practical Tuning Cheat Sheet
    - **Tuning Keyword vs. Vector Weight ($\alpha$)**:
      - **Exact codes / SKUs / compliance IDs**: Start at $\alpha = 0.25\text{–}0.40$ (favor sparse BM25/SPLADE).
      - **Balanced technical docs & Q&A**: Start at $\alpha = 0.50\text{–}0.70$ (e.g., `CONVEX-ALPHA-70`: 70% vector + 30% keyword).
      - **Abstract conceptual / paraphrased queries**: Shift to $\alpha = 0.75\text{–}0.90$.
    - **Adding a Cross-Encoder Reranker**:
      - Use broad Stage-1 recall (`Top 25–50` chunks) across your parallel engines, then pass those pairs into a Cross-Encoder (such as `BAAI/bge-reranker-v2-m3`, `ms-marco-MiniLM-L-6-v2`, or **Cohere Rerank**) to filter cleanly down to the **Top 3–5** passages.
    """)


if __name__ == "__main__":
    app.run()
