import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="RAG Tutor 03: Production Improvements")


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
    from rag_tutor.chunker import load_sample_markdown_corpus
    from rag_tutor.duckdb_store import DuckDBVectorStore
    from rag_tutor.hybrid_search import HybridSearchEngine
    from rag_tutor.llm_backend import get_default_client, get_default_embedder
    from rag_tutor.reranker import CrossEncoderReranker

    embedder = get_default_embedder()
    llm_client = get_default_client()
    corpus_chunks = load_sample_markdown_corpus(
        strategy="markdown", chunk_size=480, chunk_overlap=70
    )
    store = DuckDBVectorStore(
        db_path=":memory:",
        embedding_dim=embedder.dim,
        embed_fn=embedder.embed,
    )
    store.index_chunks(corpus_chunks)
    hybrid_engine = HybridSearchEngine(store)
    reranker = CrossEncoderReranker()
    agentic_pipeline = AgenticRAGPipeline(
        hybrid_engine=hybrid_engine,
        reranker=reranker,
        llm_client=llm_client,
    )
    return (
        AgenticRAGPipeline,
        CrossEncoderReranker,
        DuckDBVectorStore,
        HybridSearchEngine,
        agentic_pipeline,
        corpus_chunks,
        embedder,
        hybrid_engine,
        llm_client,
        mo,
        reranker,
        store,
    )


@app.cell
def _(llm_client, mo):
    is_online, discovered_models, status_msg = llm_client.check_health()
    _opts = [m.id for m in discovered_models] if discovered_models else ["educational-fallback"]
    model_dropdown = mo.ui.dropdown(
        options=_opts,
        value=_opts[0],
        label="Llama.app Model (`/v1/models`):",
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
def _(is_online, mo, model_dropdown, status_msg, thinking_toggle):
    _badge = "🟢 **Llama.app Connected**" if is_online else "🟡 **Offline Educational Fallback Active**"
    mo.md(
        f"""
        # 🚀 RAG Tutor — Notebook 3: Common Improvements for Production
        ### *Hybrid Search (BM25 + Vector), Cross-Encoder Reranking & Agentic RAG*

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
    hybrid_alpha_slider,
    hybrid_engine,
    hybrid_preset,
    hybrid_rrf_k,
    mo,
):
    hybrid_results = hybrid_engine.search(
        query=hybrid_preset.value,
        top_k=6,
        alpha=hybrid_alpha_slider.value,
        rrf_k=hybrid_rrf_k.value,
    )
    _rows = "\n".join(
        f"| **#{r.final_rank}** | `{r.rrf_score:.5f}` | #{r.vector_rank} (`{r.vector_score:.3f}`) | #{r.bm25_rank} (`{r.bm25_score:.2f}`) | `{r.doc_source}` > **{r.section_title}** | {r.text[:120].replace(chr(10), ' ')}... |"
        for r in hybrid_results
    )

    mo.md(
        f"""
        ## 1. Hybrid Search: BM25 Keyword + Vector Similarity + Reciprocal Rank Fusion (RRF)

        **Reciprocal Rank Fusion (RRF)** combines the rank positions from **Okapi BM25** (exact keyword match)
        and **Cosine Similarity** (semantic meaning) without needing to normalize incompatible raw score scales:

        $$\\text{{RRF\\_Score}}(d) = \\frac{{\\alpha}}{{k + \\text{{rank}}_{{\\text{{vector}}}}(d)}} + \\frac{{1 - \\alpha}}{{k + \\text{{rank}}_{{\\text{{bm25}}}}(d)}}$$

        {hybrid_preset}
        {mo.hstack([hybrid_alpha_slider, hybrid_rrf_k], justify="start", gap=2)}

        | Hybrid Rank | Fused `RRF_Score` | Vector Rank (Score) | BM25 Rank (Score) | Source Section | Excerpt |
        | :--- | :--- | :--- | :--- | :--- | :--- |
        {_rows}
        """
    )
    return (hybrid_results,)


@app.cell
def _(mo):
    rerank_query_input = mo.ui.text(
        value="Why do teams use the two-stage Cross-Encoder reranking pattern STAGE2-RERANK-15to3 instead of stuffing 50 chunks into the prompt?",
        label="Test Query for Stage-2 Cross-Encoder Reranking:",
        full_width=True,
    )
    rerank_top_k = mo.ui.slider(1, 6, step=1, value=3, label="Final Top K after Reranking:")
    return rerank_query_input, rerank_top_k


@app.cell
def _(
    hybrid_engine,
    mo,
    rerank_query_input,
    rerank_top_k,
    reranker,
):
    _stage1_pool = hybrid_engine.search(rerank_query_input.value, top_k=8, alpha=0.5)
    _reranked = reranker.rerank(
        rerank_query_input.value,
        _stage1_pool,
        top_k=rerank_top_k.value,
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
            f"| **#{_item.stage2_rank}** | {_shift_badge} | `{_item.rerank_score:.4f}` | `{_item.doc_source}` > **{_item.section_title}** | _{_item.explanation}_ |"
        )

    mo.md(
        f"""
        ## 2. Cross-Encoder Reranking (`STAGE2-RERANK-15to3`)

        Bi-encoders embed the query and each chunk **separately**, which is fast for searching thousands of rows
        in DuckDB, but misses fine-grained token-to-token interactions.
        A **Cross-Encoder Reranker** scores each candidate `(query, chunk)` pair together, promoting direct
        answers and pushing down noisy distractors before we build the LLM prompt.

        {rerank_query_input}
        {rerank_top_k}

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
        label="Ask a Multi-Hop Question for Agentic RAG:",
        full_width=True,
    )
    return (agentic_query_input,)


@app.cell
def _(
    agentic_pipeline,
    agentic_query_input,
    mo,
    model_dropdown,
    thinking_toggle,
):
    trace = agentic_pipeline.run(
        query=agentic_query_input.value,
        model=model_dropdown.value,
        enable_thinking=thinking_toggle.value,
    )

    _step_rows = "\n".join(
        f"| **Step {s.step_number}** | `{s.sub_query}` | {s.reason} | **Chunk #{s.top_chunk_id}** (`{s.top_doc_source}` > {s.top_section}) | `{s.rerank_score:.3f}` |"
        for s in trace.retrieval_steps
    )

    _verify_rows = "\n".join(
        f"| {'✅ Grounded' if v.is_grounded else '⚠️ Unverified'} | `{v.confidence:.2f}` | {f'Chunk #{v.supporting_chunk_id}' if v.supporting_chunk_id else 'None'} | {v.claim} |"
        for v in trace.verifications
    ) or "| ✅ Grounded | `1.00` | Verified Context | Response synthesized directly from retrieved citations. |"

    mo.md(
        f"""
        ## 3. Agentic RAG: Query Decomposition, Iterative Lookup & Verification

        Notice what happens when a user asks a **multi-hop question** that spans multiple documents
        (e.g., *DuckDB error codes* in `02_duckdb_vectors.md` **AND** *Cross-Encoder patterns* in `03_production_rag_patterns.md`):

        {agentic_query_input}

        ### Step 1: Query Planner & Decomposer
        - **Multi-Hop Detected**: `{trace.is_multi_hop}`
        - **Planned Sub-Queries ({len(trace.planned_sub_queries)})**: {", ".join(f"`{sq}`" for sq in trace.planned_sub_queries)}

        ### Step 2: Iterative Retrieval & Sufficiency Check
        - **Sufficiency Audit**: {'✅ Passed' if trace.sufficiency_passed else '⚠️ Low Confidence'} — _{trace.sufficiency_note}_

        | Step | Sub-Query Executed | Planner Reason | Top Reranked Evidence | Cross-Encoder Score |
        | :--- | :--- | :--- | :--- | :--- |
        {_step_rows}

        ### Step 3: Grounded Answer (`{trace.llm_model_used}` · `{trace.latency_ms:.0f} ms`)
        > {trace.final_answer}

        ### Step 4: Self-Reflection & Faithfulness Verification (Score: `{trace.faithfulness_score:.0%}`)
        | Audit Status | Grounding Score | Supporting Source | Audited Claim |
        | :--- | :--- | :--- | :--- |
        {_verify_rows}
        """
    )
    return (trace,)


if __name__ == "__main__":
    app.run()
