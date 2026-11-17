import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="RAG Tutor 03: Production Improvements")


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
        AgenticRAGPipeline,
        CrossEncoderReranker,
        DuckDBVectorStore,
        HybridSearchEngine,
        build_agentic_mermaid_diagram,
        get_default_client,
        load_sample_markdown_corpus,
        render_cross_attention_heatmap_svg,
        render_faithfulness_audit_svg,
        render_hybrid_rrf_svg,
        render_rerank_slopegraph_svg,
    )

    llm_client = get_default_client()
    corpus_chunks = load_sample_markdown_corpus(
        strategy="markdown", chunk_size=480, chunk_overlap=70
    )
    store = DuckDBVectorStore()
    store.index_chunks(corpus_chunks)
    hybrid_engine = HybridSearchEngine(store)
    reranker = CrossEncoderReranker()
    agentic_pipeline = AgenticRAGPipeline(
        hybrid_engine=hybrid_engine,
        reranker=reranker,
        llm_client=llm_client,
    )
    return (
        agentic_pipeline,
        build_agentic_mermaid_diagram,
        html,
        hybrid_engine,
        llm_client,
        mo,
        render_cross_attention_heatmap_svg,
        render_faithfulness_audit_svg,
        render_hybrid_rrf_svg,
        render_rerank_slopegraph_svg,
        reranker,
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
def _(is_online, mo, model_dropdown, status_msg, thinking_toggle):
    _badge = "🟢 **Llama.app Connected**" if is_online else "🟡 **Offline Educational Fallback Active**"
    mo.md(
        f"""
        # 🚀 RAG Tutor — Notebook 3: Common Improvements for Production
        ### *Visualizing Hybrid Search (RRF), Cross-Encoder Slopegraphs & Agentic RAG*

        A basic 50-line vector RAG works great for simple demos, but in production systems it runs into three classic failure modes:
        1. **Exact Keyword Blindness**: Dense vectors compress a whole passage into one vector and often miss exact error codes, part numbers, or function names (solved by **Hybrid Search**).
        2. **Noisy Bi-Encoder Distractors**: Chunks that mention the same general topic without answering the specific question sneak into the Top $K$ (solved by **Cross-Encoder Reranking**).
        3. **Multi-Hop & Unverified Answers**: Single-shot retrieval fails when a user asks a multi-part question spanning multiple documents (solved by **Agentic RAG**).

        ---
        **Local LLM Backend**: {_badge} — _{status_msg}_
        {mo.hstack([model_dropdown, thinking_toggle], justify="start", gap=2)}
        """
    )
    return


@app.cell
def _(mo):
    hybrid_preset = mo.ui.dropdown(
        options={
            "Exact Code Lookup: ERR-VEC-768 and BM25-K1-1.2": "What does diagnostic code ERR-VEC-768 mean and what is BM25-K1-1.2?",
            "Conceptual Query: Why do language models make things up?": "Why do language models fabricate answers when asked about internal data?",
            "RRF Formula & Config: How does RRF-K60 merge rankings?": "How does Reciprocal Rank Fusion RRF-K60 combine lexical and vector ranks?",
        },
        value="Exact Code Lookup: ERR-VEC-768 and BM25-K1-1.2",
        label="Preset Query:",
    )
    hybrid_alpha_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.1,
        value=0.5,
        label="Fusion Weight α (0.0 = Pure BM25 Keyword, 0.5 = Balanced Hybrid, 1.0 = Pure Vector):",
    )
    hybrid_rrf_k = mo.ui.slider(
        start=10,
        stop=100,
        step=10,
        value=60,
        label="RRF Smoothing Constant k (`RRF-K60`):",
    )
    return hybrid_alpha_slider, hybrid_preset, hybrid_rrf_k


@app.cell
def _(
    html,
    hybrid_alpha_slider,
    hybrid_engine,
    hybrid_preset,
    hybrid_rrf_k,
    mo,
    render_hybrid_rrf_svg,
):
    _hybrid_results = hybrid_engine.search(
        query=hybrid_preset.value,
        top_k=6,
        alpha=hybrid_alpha_slider.value,
        rrf_k=hybrid_rrf_k.value,
    )
    _rrf_svg = render_hybrid_rrf_svg(
        hybrid_results=_hybrid_results,
        alpha=hybrid_alpha_slider.value,
        rrf_k=hybrid_rrf_k.value,
    )

    _rows = "\n".join(
        f"| **#{r.final_rank}** | `{r.rrf_score:.5f}` | #{r.vector_rank} (`{r.vector_score:.3f}`) | #{r.bm25_rank} (`{r.bm25_score:.2f}`) | `{html.escape(r.doc_source)}` > **{html.escape(r.section_title).replace('|', '\\|')}** | {html.escape(r.text[:110].replace(chr(10), ' ')).replace('|', '\\|')}... |"
        for r in _hybrid_results
    )

    _mode_lbl = (
        "Pure BM25 Lexical"
        if hybrid_alpha_slider.value == 0.0
        else ("Pure Vector Semantic" if hybrid_alpha_slider.value == 1.0 else f"Balanced Hybrid (α={hybrid_alpha_slider.value:.1f})")
    )
    if _hybrid_results:
        _top = _hybrid_results[0]
        _stats = mo.hstack(
            [
                mo.stat(
                    value=f"Chunk #{_top.chunk_id}",
                    label="Hybrid Rank #1 Winner",
                    caption=f"{_top.doc_source} > {_top.section_title}",
                    bordered=True,
                ),
                mo.stat(
                    value=f"Vec #{_top.vector_rank} / BM25 #{_top.bm25_rank}",
                    label="Winner's Retriever Ranks",
                    caption=f"Fused RRF = {_top.rrf_score:.5f}",
                    bordered=True,
                ),
                mo.stat(
                    value=_mode_lbl,
                    label="Active Fusion Mode",
                    caption=f"Smoothing constant k = {hybrid_rrf_k.value}",
                    bordered=True,
                ),
            ],
            widths="equal",
            gap=2,
        )
    else:
        _stats = mo.md("_No matching chunks._")

    mo.md(
        f"""
        ## 1. Hybrid Search: BM25 Keyword + Vector Similarity + Reciprocal Rank Fusion (RRF)

        **Reciprocal Rank Fusion (RRF)** combines the rank positions from **Okapi BM25** (exact keyword match)
        and **Cosine Similarity** (semantic meaning) without needing to normalize incompatible raw score scales:

        $$\\text{{RRF\\_Score}}(d) = \\frac{{\\alpha}}{{k + \\text{{rank}}_{{\\text{{vector}}}}(d)}} + \\frac{{1 - \\alpha}}{{k + \\text{{rank}}_{{\\text{{bm25}}}}(d)}}$$

        Drag **Fusion Weight $\\alpha$** and **RRF Constant $k$** below to watch the **Stacked RRF Decomposition Bars**
        (left) and the **Rank Discount Curve** (right) shift in real time:

        {hybrid_preset}
        {mo.hstack([hybrid_alpha_slider, hybrid_rrf_k], justify="start", gap=2)}

        {_stats}

        {mo.Html(_rrf_svg)}

        | Hybrid Rank | Fused `RRF_Score` | Vector Rank (Score) | BM25 Rank (Score) | Source Section | Excerpt |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        {_rows}
        """
    )
    return


@app.cell
def _(mo):
    rerank_query_input = mo.ui.text(
        value="Why do teams use the two-stage Cross-Encoder reranking pattern STAGE2-RERANK-15to3 instead of stuffing 50 chunks into the prompt?",
        label="Test Query for Stage-2 Cross-Encoder Reranking:",
        full_width=True,
    )
    rerank_top_k = mo.ui.slider(1, 6, step=1, value=3, label="Final Top K after Reranking:")
    rerank_inspect_slot = mo.ui.dropdown(
        options={
            "Inspect Stage-2 Rank #1 (Top Promoted Answer)": 0,
            "Inspect Stage-2 Rank #2": 1,
            "Inspect Demoted Distractor (Lowest Ranked in Pool)": -1,
        },
        value="Inspect Stage-2 Rank #1 (Top Promoted Answer)",
        label="Cross-Attention Matrix Target:",
    )
    return rerank_inspect_slot, rerank_query_input, rerank_top_k


@app.cell
def _(
    html,
    hybrid_engine,
    mo,
    render_cross_attention_heatmap_svg,
    render_rerank_slopegraph_svg,
    rerank_inspect_slot,
    rerank_query_input,
    rerank_top_k,
    reranker,
):
    _stage1_pool = hybrid_engine.search(rerank_query_input.value, top_k=7, alpha=0.5)
    _all_reranked = reranker.rerank(
        rerank_query_input.value,
        _stage1_pool,
        top_k=len(_stage1_pool),
    )
    _reranked = _all_reranked[: rerank_top_k.value]

    _slopegraph_html = render_rerank_slopegraph_svg(
        all_reranked=_all_reranked,
        top_k=rerank_top_k.value,
    )

    _target_idx = rerank_inspect_slot.value
    _inspected = _all_reranked[_target_idx] if _all_reranked else None
    _attn_html = (
        render_cross_attention_heatmap_svg(
            query=rerank_query_input.value,
            chunk_text=_inspected.text,
            section_title=f"Chunk #{_inspected.chunk_id}: {_inspected.section_title}",
            rerank_score=_inspected.rerank_score,
            explanation=_inspected.explanation,
        )
        if _inspected is not None
        else ""
    )

    _rerank_rows = []
    for _item in _reranked:
        if _item.rank_delta > 0:
            _shift_badge = f"🟢 **+{_item.rank_delta} Promoted** (was #{_item.stage1_rank})"
        elif _item.rank_delta < 0:
            _shift_badge = f"🔴 **{_item.rank_delta} Demoted** (was #{_item.stage1_rank})"
        else:
            _shift_badge = f"⚪ **Unchanged** (#{_item.stage1_rank})"
        _rerank_rows.append(
            f"| **#{_item.stage2_rank}** | {_shift_badge} | `{_item.rerank_score:.4f}` | `{html.escape(_item.doc_source)}` > **{html.escape(_item.section_title).replace('|', '\\|')}** | _{html.escape(_item.explanation).replace('|', '\\|')}_ |"
        )

    mo.md(
        f"""
        ## 2. Cross-Encoder Reranking (`STAGE2-RERANK-15to3`)

        Bi-encoders embed the query and each chunk **separately**, which is fast for searching thousands of rows
        in DuckDB, but misses fine-grained token-to-token interactions.
        A **Cross-Encoder Reranker** scores each candidate `(query, chunk)` pair together, promoting direct
        answers above the **Top-$K$ Cutoff Line** and pushing down noisy distractors!

        {rerank_query_input}
        {mo.hstack([rerank_top_k, rerank_inspect_slot], justify="start", gap=3)}

        {mo.Html(_slopegraph_html)}

        {mo.Html(_attn_html)}

        | Stage-2 Rank | Rank Shift from Stage-1 | Cross-Encoder Score | Document & Section | Reranker Interaction Signals |
        | :--- | :--- | :--- | :--- | :--- |
        {chr(10).join(_rerank_rows)}
        """
    )
    return


@app.cell
def _(mo):
    agentic_query_input = mo.ui.text(
        value="What causes DuckDB error ERR-VEC-768 and how does the two-stage reranking pattern STAGE2-RERANK-15to3 improve precision?",
        label="Ask a Multi-Hop Question for Agentic RAG (watch the Mermaid execution graph rewire live):",
        full_width=True,
    )
    return (agentic_query_input,)


@app.cell
def _(
    agentic_pipeline,
    agentic_query_input,
    build_agentic_mermaid_diagram,
    html,
    mo,
    model_dropdown,
    render_faithfulness_audit_svg,
    thinking_toggle,
):
    _trace = agentic_pipeline.run(
        query=agentic_query_input.value,
        model=model_dropdown.value,
        enable_thinking=thinking_toggle.value,
    )

    _mermaid_diagram = mo.mermaid(build_agentic_mermaid_diagram(_trace))
    _audit_html = render_faithfulness_audit_svg(_trace)

    _step_rows = "\n".join(
        f"| **Step {s.step_number}** | `{html.escape(s.sub_query).replace('|', '\\|')}` | {html.escape(s.reason)} | **Chunk #{s.top_chunk_id}** (`{html.escape(s.top_doc_source)}` > {html.escape(s.top_section).replace('|', '\\|')}) | `{s.rerank_score:.3f}` |"
        for s in _trace.retrieval_steps
    )

    _trace_tree = mo.tree(
        {
            "original_query": _trace.original_query,
            "is_multi_hop": _trace.is_multi_hop,
            "planned_sub_queries": _trace.planned_sub_queries,
            "sufficiency_passed": _trace.sufficiency_passed,
            "faithfulness_score": _trace.faithfulness_score,
            "retrieved_chunks": [
                f"Chunk #{c.chunk_id} ({c.doc_source} > {c.section_title}) [score={c.rerank_score:.3f}]"
                for c in _trace.final_context_chunks
            ],
        },
        label="AgenticTrace",
    )

    mo.md(
        f"""
        ## 3. Agentic RAG: Live Execution Graph, Iterative Lookup & Verification

        Notice what happens when a user asks a **multi-hop question** that spans multiple documents
        (e.g., *DuckDB error codes* in `02_duckdb_vectors.md` **AND** *Cross-Encoder patterns* in `03_production_rag_patterns.md`).
        Try removing `"and how..."` from the query below to see the **Mermaid Execution Graph** collapse from a multi-branch plan into a single-hop lookup!

        {agentic_query_input}

        ### Live Agentic Execution Flowchart (`mo.mermaid`)
        {_mermaid_diagram}

        ### Step 1 & 2: Sub-Query Planner & Iterative Retrieval
        - **Multi-Hop Detected**: `{_trace.is_multi_hop}` · **Planned Sub-Queries ({len(_trace.planned_sub_queries)})**: {", ".join(f"`{html.escape(sq)}`" for sq in _trace.planned_sub_queries)}
        - **Sufficiency Audit**: {'✅ Passed' if _trace.sufficiency_passed else '⚠️ Low Confidence'} — _{html.escape(_trace.sufficiency_note)}_

        | Step | Sub-Query Executed | Planner Reason | Top Reranked Evidence | Cross-Encoder Score |
        | :--- | :--- | :--- | :--- | :--- |
        {_step_rows}

        ### Step 3: Grounded Answer (`{_trace.llm_model_used}` · `{_trace.latency_ms:.0f} ms`)
        > {html.escape(_trace.final_answer)}

        ### Step 4: Self-Reflection & Faithfulness Verification
        {mo.Html(_audit_html)}

        {mo.accordion({"🌳 Inspect Raw AgenticTrace Object (`mo.tree`)": _trace_tree})}
        """
    )
    return


if __name__ == "__main__":
    app.run()
