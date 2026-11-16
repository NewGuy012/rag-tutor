import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="RAG Tutor 04: Capstone Final Project Studio")


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo as mo

    _repo_root = Path(__file__).resolve().parents[1]
    _src_dir = _repo_root / "src"
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    from rag_tutor.agentic_rag import AgenticRAGPipeline
    from rag_tutor.chunker import (
        chunk_fixed_character,
        chunk_markdown_sections,
        load_sample_markdown_corpus,
    )
    from rag_tutor.duckdb_store import DuckDBVectorStore
    from rag_tutor.hybrid_search import HybridSearchEngine
    from rag_tutor.llm_backend import get_default_client, get_default_embedder
    from rag_tutor.reranker import CrossEncoderReranker
    from rag_tutor.scratch_rag import SimpleVectorDB, build_rag_prompt

    embedder = get_default_embedder()
    llm_client = get_default_client()
    base_corpus = load_sample_markdown_corpus(
        strategy="markdown", chunk_size=480, chunk_overlap=70
    )
    return (
        AgenticRAGPipeline,
        CrossEncoderReranker,
        DuckDBVectorStore,
        HybridSearchEngine,
        Path,
        SimpleVectorDB,
        base_corpus,
        build_rag_prompt,
        chunk_fixed_character,
        chunk_markdown_sections,
        embedder,
        llm_client,
        mo,
    )


@app.cell
def _(llm_client, mo):
    is_online, discovered_models, status_msg = llm_client.check_health()
    _opts = [m.id for m in discovered_models] if discovered_models else ["educational-fallback"]
    model_dropdown = mo.ui.dropdown(
        options=_opts,
        value=_opts[0],
        label="Llama.app Model:",
    )
    thinking_toggle = mo.ui.switch(value=False, label="Enable Thinking")
    return (
        discovered_models,
        is_online,
        model_dropdown,
        status_msg,
        thinking_toggle,
    )


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
        options={"Markdown Section-Aware": "markdown", "Fixed-Character Window": "fixed"},
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
    AgenticRAGPipeline,
    CrossEncoderReranker,
    DuckDBVectorStore,
    HybridSearchEngine,
    SimpleVectorDB,
    base_corpus,
    build_rag_prompt,
    chunk_fixed_character,
    chunk_markdown_sections,
    chunk_strategy_ctl,
    custom_doc_editor,
    embedder,
    hybrid_alpha_ctl,
    is_online,
    llm_client,
    mo,
    model_dropdown,
     status_msg,
    studio_query_input,
    thinking_toggle,
    top_k_ctl,
    use_agentic_ctl,
    use_reranker_ctl,
):
    # 1. Chunk the user's custom document and combine with the core sample docs
    if chunk_strategy_ctl.value == "fixed":
        _custom_chunks = chunk_fixed_character(
            custom_doc_editor.value,
            chunk_size=450,
            chunk_overlap=60,
            doc_source="custom_runbook.md",
        )
    else:
        _custom_chunks = chunk_markdown_sections(
            custom_doc_editor.value,
            max_chunk_size=480,
            chunk_overlap=60,
            doc_source="custom_runbook.md",
        )

    _combined = list(base_corpus)
    _next_id = len(_combined) + 1
    for _c in _custom_chunks:
        _c.chunk_id = _next_id
        _next_id += 1
        _combined.append(_c)

    # 2. Baseline Pipeline A: Scratch In-Memory Vector RAG (Notebook 1 style)
    _scratch_db = SimpleVectorDB()
    for _c in _combined:
        _scratch_db.add_chunk(_c.text, embedder.embed(_c.text))
    _q_vec = embedder.embed(studio_query_input.value)
    _baseline_hits = _scratch_db.retrieve(_q_vec, top_n=top_k_ctl.value)
    _base_sys, _base_usr = build_rag_prompt(studio_query_input.value, _baseline_hits)
    _base_ans = llm_client.generate(
        system_prompt=_base_sys,
        user_prompt=_base_usr,
        model=model_dropdown.value,
        enable_thinking=thinking_toggle.value,
        retrieved_context=[t for t, _ in _baseline_hits],
    )

    # 3. Production Pipeline B: DuckDB + Hybrid RRF + Reranker + Agentic Workflow
    _store = DuckDBVectorStore(
        db_path=":memory:",
        embedding_dim=embedder.dim,
        embed_fn=embedder.embed,
    )
    _store.index_chunks(_combined)
    _hybrid = HybridSearchEngine(_store)
    _reranker = CrossEncoderReranker()

    if use_agentic_ctl.value:
        _agent = AgenticRAGPipeline(_hybrid, _reranker, llm_client)
        _trace = _agent.run(
            query=studio_query_input.value,
            model=model_dropdown.value,
            enable_thinking=thinking_toggle.value,
            final_top_k=top_k_ctl.value,
        )
        _prod_answer = _trace.final_answer
        _prod_chunks = [
            (f"Chunk #{c.chunk_id} ({c.doc_source} > {c.section_title})", c.rerank_score, c.text)
            for c in _trace.final_context_chunks
        ]
        _prod_meta = (
            f"Sub-queries: {len(_trace.planned_sub_queries)} · "
            f"Faithfulness: {_trace.faithfulness_score:.0%}"
        )
    else:
        _s1 = _hybrid.search(
            studio_query_input.value,
            top_k=max(10, top_k_ctl.value * 3),
            alpha=hybrid_alpha_ctl.value,
        )
        if use_reranker_ctl.value:
            _s2 = _reranker.rerank(studio_query_input.value, _s1, top_k=top_k_ctl.value)
            _prod_chunks = [
                (f"Chunk #{c.chunk_id} ({c.doc_source} > {c.section_title})", c.rerank_score, c.text)
                for c in _s2
            ]
        else:
            _prod_chunks = [
                (f"Chunk #{c.chunk_id} ({c.doc_source} > {c.section_title})", c.rrf_score, c.text)
                for c in _s1[: top_k_ctl.value]
            ]
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

    _base_chunks_md = "\n".join(
        f"- **#{i + 1}** (`cos={sc:.3f}`): {txt[:120].replace(chr(10), ' ')}..."
        for i, (txt, sc) in enumerate(_baseline_hits)
    )
    _prod_chunks_md = "\n".join(
        f"- **#{i + 1}** (`score={sc:.3f}` · {lbl}): {txt[:120].replace(chr(10), ' ')}..."
        for i, (lbl, sc, txt) in enumerate(_prod_chunks)
    )

    _badge = "🟢 **Llama.app Connected**" if is_online else "🟡 **Offline Educational Fallback Active**"

    mo.md(
        f"""
        # 🎓 RAG Tutor — Notebook 4: Capstone Final Project Studio
        ### *Bring Your Own Documents & Compare Basic vs. Production RAG Side-by-Side*

        Congratulations on reaching the **Final Project**! Here you can edit or paste your own custom
        Markdown knowledge base, configure every stage of your DuckDB + `Llama.app` RAG pipeline, and
        compare **Pipeline A (Basic 50-Line Scratch RAG)** against **Pipeline B (Production RAG)** in real time.

        **Backend**: {_badge} — _{status_msg}_
        {mo.hstack([model_dropdown, thinking_toggle], justify="start", gap=2)}

        ---
        ## 1. Knowledge Base Editor (`{len(_combined)}` Total Chunks Indexed in DuckDB)
        {custom_doc_editor}

        ## 2. Production Pipeline Controls
        {mo.hstack([chunk_strategy_ctl, hybrid_alpha_ctl, top_k_ctl], justify="start", gap=2)}
        {mo.hstack([use_reranker_ctl, use_agentic_ctl], justify="start", gap=3)}

        ## 3. A/B Pipeline Arena
        {studio_query_input}

        | Metric / Stage | Pipeline A: Basic Scratch RAG (Pure Vector) | Pipeline B: Production RAG ({_prod_meta}) |
        | :--- | :--- | :--- |
        | **Storage Engine** | Python `list[tuple[str, list[float]]]` | Embedded **DuckDB** (`FLOAT[{embedder.dim}]` + BM25) |
        | **Retrieved Context** | {_base_chunks_md} | {_prod_chunks_md} |
        | **Generated Answer** | {_base_ans.content} | {_prod_answer} |

        ---
        ## 🏆 Guided Capstone Challenges
        1. **Challenge 1 (Exact Identifier Rescue)**: Set **Agentic Multi-Hop** to `OFF`, **Cross-Encoder Reranker** to `OFF`, and slide **Hybrid α** from `1.0` (Pure Vector) down to `0.0` (Pure BM25) while querying `"INC-APOLLO-99"`. Watch how lexical BM25 boosts exact incident codes!
        2. **Challenge 2 (Add Your Own Knowledge)**: Add a new section `## 3. Team On-Call Schedule` to the Markdown editor above with a specific engineer's name and pager code, then ask your pipeline who is on call!
        3. **Challenge 3 (Multi-Hop Synthesis)**: Enable **Agentic Multi-Hop & Verification** and ask a question that combines a fact from your custom runbook with a fact from `02_duckdb_vectors.md`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
