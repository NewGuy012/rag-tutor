import marimo

__generated_with = "0.10.0"
app = marimo.App(
    width="medium", app_title="RAG Tutor 04: Capstone Final Project Studio"
)


@app.cell
def _():
    import dataclasses
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
        SimpleVectorDB,
        build_rag_prompt,
        build_studio_pipeline_mermaid,
        chunk_document,
        get_default_client,
        get_default_embedder,
        load_sample_markdown_corpus,
        render_ab_arena_comparison_svg,
        render_chunk_timeline_svg,
    )

    embedder = get_default_embedder()
    llm_client = get_default_client()
    base_corpus = load_sample_markdown_corpus(
        strategy="markdown", chunk_size=480, chunk_overlap=70
    )
    duck_store = DuckDBVectorStore()
    hybrid_engine = HybridSearchEngine(duck_store)
    reranker = CrossEncoderReranker()
    return (
        AgenticRAGPipeline,
        SimpleVectorDB,
        base_corpus,
        build_rag_prompt,
        build_studio_pipeline_mermaid,
        chunk_document,
        dataclasses,
        duck_store,
        embedder,
        html,
        hybrid_engine,
        llm_client,
        mo,
        render_ab_arena_comparison_svg,
        render_chunk_timeline_svg,
        reranker,
    )


@app.cell
def _(llm_client, mo):
    is_online, _models, status_msg = llm_client.check_health()
    _opts = _models or ["educational-fallback"]
    model_dropdown = mo.ui.dropdown(
        options=_opts,
        value=_opts[0],
        label="Llama.app Model:",
    )
    thinking_toggle = mo.ui.switch(value=False, label="Enable Thinking")
    return is_online, model_dropdown, status_msg, thinking_toggle


@app.cell
def _(mo):
    custom_doc_editor = mo.ui.text_area(
        value=(
            "# Project Apollo Internal Runbook\n\n"
            "## 1. Service Architecture & Port Binding\n"
            "The Apollo Inference Gateway runs locally on macOS port `9931` via `Llama.app` "
            "and stores vector embeddings inside `apollo_vectors.duckdb` using 128-dimensional "
            "`FLOAT[128]` arrays.\n\n"
            "## 2. Incident Recovery Code `INC-APOLLO-99`\n"
            "If retrieval latency exceeds 120ms during peak traffic, engineers must follow "
            "runbook code `INC-APOLLO-99`: reduce Stage-1 candidate pool from 50 to 15 and "
            "enable Cross-Encoder reranking (`STAGE2-RERANK-15to3`) with Top-3 cutoff.\n"
        ),
        label="Bring Your Own Markdown Document (added live to the DuckDB knowledge base):",
        rows=8,
        full_width=True,
    )

    chunk_strategy_ctl = mo.ui.dropdown(
        options={
            "Markdown Section-Aware": "markdown",
            "Fixed-Character Window": "fixed",
        },
        value="Markdown Section-Aware",
        label="Chunking Strategy:",
    )
    hybrid_alpha_ctl = mo.ui.slider(
        0.0, 1.0, step=0.1, value=0.5, label="Hybrid α (BM25 ↔ Vector):"
    )
    use_reranker_ctl = mo.ui.switch(value=True, label="Cross-Encoder Reranker")
    use_agentic_ctl = mo.ui.switch(value=True, label="Agentic Multi-Hop & Verification")
    top_k_ctl = mo.ui.slider(1, 5, step=1, value=3, label="Final Top K Context Chunks:")

    studio_query_input = mo.ui.text(
        value="What should engineers do for incident code INC-APOLLO-99 and how does it relate to STAGE2-RERANK-15to3?",
        label="Ask Your Capstone RAG Pipeline:",
        full_width=True,
    )
    return (
        chunk_strategy_ctl,
        custom_doc_editor,
        hybrid_alpha_ctl,
        studio_query_input,
        top_k_ctl,
        use_agentic_ctl,
        use_reranker_ctl,
    )


@app.cell
def _(
    SimpleVectorDB,
    base_corpus,
    chunk_document,
    chunk_strategy_ctl,
    custom_doc_editor,
    dataclasses,
    duck_store,
    embedder,
    render_chunk_timeline_svg,
):
    _target_size = 450 if chunk_strategy_ctl.value == "fixed" else 480
    custom_chunks = chunk_document(
        custom_doc_editor.value,
        strategy=chunk_strategy_ctl.value,
        chunk_size=_target_size,
        chunk_overlap=60,
        doc_source="custom_runbook.md",
    )
    custom_timeline_svg = render_chunk_timeline_svg(
        raw_text=custom_doc_editor.value,
        chunks=custom_chunks,
        strategy_name=chunk_strategy_ctl.value,
        target_chunk_size=_target_size,
    )
    combined_chunks = [
        *base_corpus,
        *(
            dataclasses.replace(_c, chunk_id=len(base_corpus) + _i + 1)
            for _i, _c in enumerate(custom_chunks)
        ),
    ]
    scratch_db = SimpleVectorDB()
    scratch_db.index_chunks([_c.text for _c in combined_chunks], embedder.embed)
    duck_store.index_chunks(combined_chunks, replace=True)
    return combined_chunks, custom_chunks, custom_timeline_svg, scratch_db


@app.cell
def _(
    AgenticRAGPipeline,
    base_corpus,
    build_rag_prompt,
    build_studio_pipeline_mermaid,
    chunk_strategy_ctl,
    combined_chunks,
    custom_chunks,
    custom_doc_editor,
    custom_timeline_svg,
    embedder,
    html,
    hybrid_alpha_ctl,
    hybrid_engine,
    is_online,
    llm_client,
    mo,
    model_dropdown,
    render_ab_arena_comparison_svg,
    reranker,
    scratch_db,
    status_msg,
    studio_query_input,
    thinking_toggle,
    top_k_ctl,
    use_agentic_ctl,
    use_reranker_ctl,
):
    def _to_prod_chunks(items) -> list[tuple[str, float, str]]:
        return [
            (
                f"Chunk #{c.chunk_id} ({c.doc_source} > {c.section_title})",
                float(
                    c.rerank_score
                    if hasattr(c, "rerank_score")
                    else c.rrf_score
                ),
                c.text,
            )
            for c in items
        ]

    _query_clean = studio_query_input.value.strip()
    _q_vec = embedder.embed(_query_clean)
    _baseline_hits = (
        scratch_db.retrieve(_q_vec, top_n=top_k_ctl.value) if _query_clean else []
    )
    _base_sys, _base_usr = build_rag_prompt(_query_clean, _baseline_hits)
    _base_ans = llm_client.generate(
        system_prompt=_base_sys,
        user_prompt=_base_usr,
        model=model_dropdown.value,
        enable_thinking=thinking_toggle.value,
        retrieved_context=[t for t, _ in _baseline_hits],
    )

    if use_agentic_ctl.value:
        _agent = AgenticRAGPipeline(hybrid_engine, reranker, llm_client)
        _trace = _agent.run(
            query=studio_query_input.value,
            model=model_dropdown.value,
            enable_thinking=thinking_toggle.value,
            final_top_k=top_k_ctl.value,
            alpha=hybrid_alpha_ctl.value,
        )
        _prod_answer = _trace.final_answer
        _prod_chunks = _to_prod_chunks(_trace.final_context_chunks)
        _prod_meta = (
            f"Hybrid α={hybrid_alpha_ctl.value:.1f} · "
            f"Sub-queries: {len(_trace.planned_sub_queries)} · "
            f"Faithfulness: {_trace.faithfulness_score:.0%}"
        )
    else:
        _s1 = hybrid_engine.search(
            studio_query_input.value,
            top_k=max(10, top_k_ctl.value * 3),
            alpha=hybrid_alpha_ctl.value,
        )
        _selected = (
            reranker.rerank(studio_query_input.value, _s1, top_k=top_k_ctl.value)
            if use_reranker_ctl.value
            else _s1[: top_k_ctl.value]
        )
        _prod_chunks = _to_prod_chunks(_selected)
        _prod_sys, _prod_usr = build_rag_prompt(
            studio_query_input.value,
            [(txt, sc) for _, sc, txt in _prod_chunks],
        )
        _prod_gen = llm_client.generate(
            system_prompt=_prod_sys,
            user_prompt=_prod_usr,
            model=model_dropdown.value,
            enable_thinking=thinking_toggle.value,
            retrieved_context=[txt for _, _, txt in _prod_chunks],
        )
        _prod_answer = _prod_gen.content
        _prod_meta = f"Hybrid α={hybrid_alpha_ctl.value:.1f} · Reranker={'ON' if use_reranker_ctl.value else 'OFF'}"

    _pipeline_mermaid = mo.mermaid(
        build_studio_pipeline_mermaid(
            chunk_strategy=chunk_strategy_ctl.value,
            alpha=hybrid_alpha_ctl.value,
            use_reranker=use_reranker_ctl.value,
            use_agentic=use_agentic_ctl.value,
            top_k=top_k_ctl.value,
            total_chunks=len(combined_chunks),
        )
    )

    _ab_bars_html = render_ab_arena_comparison_svg(
        baseline_hits=_baseline_hits,
        prod_chunks=_prod_chunks,
    )

    _base_peak = _baseline_hits[0][1] if _baseline_hits else 0.0
    _prod_peak = _prod_chunks[0][1] if _prod_chunks else 0.0

    _kpis = mo.hstack(
        [
            mo.stat(
                value=f"{len(combined_chunks)} Chunks",
                label="DuckDB Knowledge Base",
                caption=f"{len(base_corpus)} core + {len(custom_chunks)} custom chunks",
                bordered=True,
            ),
            mo.stat(
                value=f"cos = {_base_peak:.3f}",
                label="Pipeline A Peak Score",
                caption="Pure-Python Scratch Vector RAG",
                bordered=True,
            ),
            mo.stat(
                value=f"score = {_prod_peak:.3f}",
                label="Pipeline B Peak Score",
                caption=_prod_meta,
                bordered=True,
            ),
        ],
        widths="equal",
        gap=2,
    )

    _card_a = mo.callout(
        mo.vstack(
            [
                mo.md(
                    "**📘 Pipeline A: Basic 50-Line Scratch RAG**\n\n"
                    "*Storage:* Python `list` · *Retrieval:* Pure Cosine Similarity"
                ),
                mo.Html(
                    f'<div style="white-space:pre-wrap;">{html.escape(_base_ans.content)}</div>'
                ),
            ]
        ),
        kind="info",
    )
    _card_b = mo.callout(
        mo.vstack(
            [
                mo.md(
                    f"**🚀 Pipeline B: Production RAG ({_prod_meta})**\n\n"
                    f"*Storage:* Embedded DuckDB (`FLOAT[{embedder.dim}]` + BM25)"
                ),
                mo.Html(
                    f'<div style="white-space:pre-wrap;">{html.escape(_prod_answer)}</div>'
                ),
            ]
        ),
        kind="success",
    )

    _studio_tabs = mo.ui.tabs(
        {
            "⚔️ Visual A/B Arena & Generated Answers": mo.vstack(
                [
                    mo.Html(_ab_bars_html),
                    mo.hstack([_card_a, _card_b], widths="equal", gap=2),
                ]
            ),
            "🏗️ Live Pipeline B Architecture Diagram": mo.vstack(
                [
                    mo.md(
                        "Toggle the switches above to watch the active production pipeline rewire live:"
                    ),
                    _pipeline_mermaid,
                ]
            ),
            "📏 Custom Document Chunking Minimap": mo.Html(custom_timeline_svg),
        }
    )

    _badge = (
        "🟢 **Llama.app Connected**"
        if is_online
        else "🟡 **Offline Educational Fallback Active**"
    )

    mo.vstack(
        [
            mo.md(
                f"""
                # 🎓 RAG Tutor — Notebook 4: Capstone Final Project Studio
                ### *Bring Your Own Documents & Compare Basic vs. Production RAG Side-by-Side*

                Congratulations on reaching the **Final Project**! Here you can edit or paste your own custom
                Markdown knowledge base, configure every stage of your DuckDB + `Llama.app` RAG pipeline, and
                visually compare **Pipeline A (Basic 50-Line Scratch RAG)** against **Pipeline B (Production RAG)** in real time.

                **Backend**: {_badge} — _{html.escape(status_msg)}_
                """
            ),
            mo.hstack([model_dropdown, thinking_toggle], justify="start", gap=2),
            mo.md(
                f"---\n## 1. Knowledge Base Editor (`{len(combined_chunks)}` Total Chunks Indexed in DuckDB)"
            ),
            custom_doc_editor,
            mo.md("## 2. Production Pipeline Controls"),
            mo.hstack(
                [chunk_strategy_ctl, hybrid_alpha_ctl, top_k_ctl],
                justify="start",
                gap=2,
            ),
            mo.hstack([use_reranker_ctl, use_agentic_ctl], justify="start", gap=3),
            _kpis,
            mo.md("## 3. A/B Pipeline Arena"),
            studio_query_input,
            _studio_tabs,
            mo.md(
                """
                ---
                ## 🏆 Guided Capstone Challenges
                1. **Challenge 1 (Exact Identifier Rescue)**: Set **Agentic Multi-Hop** to `OFF`, **Cross-Encoder Reranker** to `OFF`, and slide **Hybrid α** from `1.0` (Pure Vector) down to `0.0` (Pure BM25) while querying `"INC-APOLLO-99"`. Watch how lexical BM25 boosts exact incident codes!
                2. **Challenge 2 (Add Your Own Knowledge)**: Add a new section `## 3. Team On-Call Schedule` to the Markdown editor above with a specific engineer's name and pager code, then inspect the **Custom Document Chunking Minimap** tab and ask your pipeline who is on call!
                3. **Challenge 3 (Multi-Hop Synthesis)**: Enable **Agentic Multi-Hop & Verification** and ask a question that combines a fact from your custom runbook with a fact from `02_duckdb_vectors.md`.
                """
            ),
        ]
    )


if __name__ == "__main__":
    app.run()
