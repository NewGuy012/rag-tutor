import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="RAG Tutor 02: Chunking & DuckDB Vectors")


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

    from rag_tutor.chunker import (
        chunk_document,
        load_sample_markdown_corpus,
    )
    from rag_tutor.duckdb_store import DuckDBVectorStore
    from rag_tutor.llm_backend import get_default_embedder
    from rag_tutor.visualizations import render_chunk_timeline_svg

    sample_docs_dir = _repo_root / "data" / "sample_docs"
    doc_files = {p.name: p.read_text(encoding="utf-8") for p in sorted(sample_docs_dir.glob("*.md"))}
    embedder = get_default_embedder()
    return (
        DuckDBVectorStore,
        Path,
        chunk_document,
        doc_files,
        embedder,
        html,
        load_sample_markdown_corpus,
        mo,
        render_chunk_timeline_svg,
        sample_docs_dir,
    )


@app.cell
def _(doc_files, mo):
    doc_selector = mo.ui.dropdown(
        options=list(doc_files.keys()),
        value="02_duckdb_vectors.md",
        label="Inspect Document:",
    )
    strategy_selector = mo.ui.dropdown(
        options={
            "Markdown Section-Aware Chunking (Recommended)": "markdown",
            "Fixed-Character Sliding Window": "fixed",
        },
        value="Markdown Section-Aware Chunking (Recommended)",
        label="Chunking Strategy:",
    )
    chunk_size_slider = mo.ui.slider(
        start=150,
        stop=900,
        step=50,
        value=450,
        label="Max Chunk Size (characters):",
    )
    chunk_overlap_slider = mo.ui.slider(
        start=0,
        stop=200,
        step=20,
        value=60,
        label="Chunk Overlap (characters):",
    )
    return (
        chunk_overlap_slider,
        chunk_size_slider,
        doc_selector,
        strategy_selector,
    )


@app.cell
def _(
    chunk_document,
    chunk_overlap_slider,
    chunk_size_slider,
    doc_files,
    doc_selector,
    html,
    mo,
    render_chunk_timeline_svg,
    strategy_selector,
):
    _raw_text = doc_files[doc_selector.value]
    preview_chunks = chunk_document(
        _raw_text,
        strategy=strategy_selector.value,
        chunk_size=chunk_size_slider.value,
        chunk_overlap=chunk_overlap_slider.value,
        doc_source=doc_selector.value,
    )
    _strat_label = (
        "Fixed-Character Sliding Window"
        if strategy_selector.value == "fixed"
        else "Markdown Section-Aware"
    )

    _timeline_svg = render_chunk_timeline_svg(
        raw_text=_raw_text,
        chunks=preview_chunks,
        strategy_name=_strat_label,
        target_chunk_size=chunk_size_slider.value,
    )

    _avg_len = (
        sum(c.char_length for c in preview_chunks) / max(1, len(preview_chunks))
    )
    _total_overlap = sum(c.overlap_chars for c in preview_chunks)
    _overlap_pct = (_total_overlap / max(1, len(_raw_text))) * 100.0

    _kpi_row = mo.hstack(
        [
            mo.stat(
                value=str(len(preview_chunks)),
                label="Chunks Generated",
                caption=f"Source: {doc_selector.value} ({len(_raw_text)} chars)",
                bordered=True,
            ),
            mo.stat(
                value=f"{_avg_len:.0f} chars",
                label="Average Chunk Size",
                caption=f"Target max: {chunk_size_slider.value} chars",
                bordered=True,
            ),
            mo.stat(
                value=f"{_total_overlap} chars ({_overlap_pct:.1f}%)",
                label="Total Overlap Context",
                caption=f"Window overlap = {chunk_overlap_slider.value} chars",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=2,
    )

    _palette = ["#1e293b", "#172554", "#134e4a", "#3b0764", "#3f2e1e"]
    _borders = ["#38bdf8", "#60a5fa", "#2dd4bf", "#c084fc", "#f59e0b"]
    _cards = []
    for _idx, _ch in enumerate(preview_chunks):
        _bg = _palette[_idx % len(_palette)]
        _bd = _borders[_idx % len(_borders)]
        _escaped = html.escape(_ch.text)
        _cards.append(
            f"""
            <div style="background:{_bg};border-left:4px solid {_bd};padding:10px 14px;margin-bottom:8px;border-radius:6px;color:#f1f5f9;font-size:13px;">
              <div style="font-weight:700;color:{_bd};margin-bottom:4px;">
                Chunk #{_ch.chunk_id} · Section: {html.escape(_ch.section_title)}
                <span style="float:right;font-weight:400;color:#cbd5e1;">
                  chars {_ch.start_char}..{_ch.end_char} ({_ch.char_length} chars, {_ch.word_count} words) · overlap={_ch.overlap_chars} chars
                </span>
              </div>
              <div style="white-space:pre-wrap;font-family:ui-monospace,monospace;line-height:1.45;">{_escaped}</div>
            </div>
            """
        )

    mo.md(
        f"""
        # 🦆 RAG Tutor — Notebook 2: Document Chunking & DuckDB Vector Store

        In Notebook 1, `ev-facts.txt` was already split into one neat sentence per line.
        Real-world knowledge bases contain multi-section Markdown guides, PDFs, and runbooks.
        If you embed an entire 10-page document into a single vector, its meaning becomes blurry;
        if you slice it into tiny 30-character fragments, sentences lose their surrounding context.

        ---
        ## 1. Interactive Document Chunking & Overlap Lab

        Adjust **Chunk Strategy**, **Max Chunk Size**, and **Sliding Overlap** below.
        Watch the **Document Minimap** to see how **Markdown Section-Aware Chunking** aligns cleanly with
        heading boundaries (`#`, `##`, `###`), whereas **Fixed-Character Sliding Windows** slice across sections
        and rely on **amber overlap zones** to preserve boundary context:

        {mo.hstack([doc_selector, strategy_selector], justify="start", gap=2)}
        {mo.hstack([chunk_size_slider, chunk_overlap_slider], justify="start", gap=2)}

        {_kpi_row}

        {mo.Html(_timeline_svg)}

        {mo.accordion({"📄 Expand to Inspect Individual Chunk Cards (" + str(len(preview_chunks)) + " chunks)": mo.Html("".join(_cards))}, expanded=True)}
        """
    )
    return (preview_chunks,)


@app.cell
def _(
    DuckDBVectorStore,
    chunk_overlap_slider,
    chunk_size_slider,
    embedder,
    load_sample_markdown_corpus,
    sample_docs_dir,
    strategy_selector,
):
    corpus_chunks = load_sample_markdown_corpus(
        corpus_dir=sample_docs_dir,
        strategy=strategy_selector.value,
        chunk_size=chunk_size_slider.value,
        chunk_overlap=chunk_overlap_slider.value,
    )
    duck_store = DuckDBVectorStore(
        db_path=":memory:",
        embedding_dim=embedder.dim,
        embed_fn=embedder.embed,
    )
    indexed_count = duck_store.index_chunks(corpus_chunks)
    return corpus_chunks, duck_store, indexed_count


@app.cell
def _(mo):
    duck_query_input = mo.ui.text(
        value="How do I store fixed-size float embeddings and compute cosine similarity in DuckDB?",
        label="Search the DuckDB `rag_chunks` table via `array_cosine_similarity`:",
        full_width=True,
    )
    duck_top_k = mo.ui.slider(1, 8, step=1, value=4, label="Top K (`LIMIT`):")
    sql_recipe_picker = mo.ui.dropdown(
        options={
            "Recipe 1: Chunk Summary Statistics by Document (`GROUP BY doc_source`)": (
                "SELECT doc_source,\n"
                "       COUNT(*) AS num_chunks,\n"
                "       ROUND(AVG(char_length), 1) AS avg_chars,\n"
                "       MIN(char_length) AS min_chars,\n"
                "       MAX(char_length) AS max_chars\n"
                "FROM rag_chunks\n"
                "GROUP BY doc_source\n"
                "ORDER BY doc_source;"
            ),
            "Recipe 2: Verify L2 Unit Normalization (`array_cosine_similarity(embedding, embedding)`)": (
                "SELECT chunk_id, doc_source, section_title, char_length,\n"
                "       ROUND(array_cosine_similarity(embedding, embedding), 4) AS self_cosine_norm\n"
                "FROM rag_chunks\n"
                "LIMIT 6;"
            ),
            "Recipe 3: Filter Chunks Containing Specific Diagnostic / Pattern Codes": (
                "SELECT chunk_id, doc_source, section_title, char_length\n"
                "FROM rag_chunks\n"
                "WHERE chunk_text LIKE '%ERR-%' OR chunk_text LIKE '%RRF-%' OR chunk_text LIKE '%STAGE2-%'\n"
                "ORDER BY chunk_id;"
            ),
            "Recipe 4: Pairwise Chunk Similarity Matrix (Self-Join in DuckDB SQL)": (
                "SELECT a.chunk_id AS chunk_a, b.chunk_id AS chunk_b,\n"
                "       a.section_title AS sec_a, b.section_title AS sec_b,\n"
                "       ROUND(array_cosine_similarity(a.embedding, b.embedding), 4) AS cosine_sim\n"
                "FROM rag_chunks a JOIN rag_chunks b ON a.chunk_id < b.chunk_id\n"
                "ORDER BY cosine_sim DESC\n"
                "LIMIT 6;"
            ),
        },
        value="Recipe 1: Chunk Summary Statistics by Document (`GROUP BY doc_source`)",
        label="Load a DuckDB SQL Recipe:",
    )
    return duck_query_input, duck_top_k, sql_recipe_picker


@app.cell
def _(mo, sql_recipe_picker):
    sql_playground_input = mo.ui.text_area(
        value=sql_recipe_picker.value,
        label="Interactive DuckDB SQL Playground (edit and run any SQL query against `rag_chunks`):",
        rows=5,
        full_width=True,
    )
    return (sql_playground_input,)


@app.cell
def _(
    duck_query_input,
    duck_store,
    duck_top_k,
    embedder,
    html,
    indexed_count,
    mo,
    sql_playground_input,
    sql_recipe_picker,
):
    _vec_hits = duck_store.search_vector(duck_query_input.value, top_k=duck_top_k.value)

    _bars_html = []
    for _idx, _r in enumerate(_vec_hits):
        _pct = max(5.0, min(100.0, _r.score * 100.0))
        _excerpt = html.escape(_r.text[:145].replace("\n", " "))
        _bars_html.append(
            f"""
            <div style="background:#020617;padding:9px 12px;border-radius:8px;border:1px solid #1e293b;margin-bottom:7px;">
              <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;margin-bottom:4px;">
                <span style="color:#f8fafc;font-family:monospace;">
                  <strong>#{_idx + 1}</strong> · <code>{html.escape(_r.doc_source)}</code> &gt; <strong>{html.escape(_r.section_title)}</strong> (Chunk #{_r.chunk_id})
                </span>
                <span style="color:#2dd4bf;font-family:monospace;font-weight:700;">vector_score = {_r.score:.4f}</span>
              </div>
              <div style="background:#1e293b;height:8px;border-radius:4px;overflow:hidden;margin-bottom:5px;">
                <div style="width:{_pct:.1f}%;background:linear-gradient(90deg,#0284c7,#2dd4bf);height:100%;border-radius:4px;"></div>
              </div>
              <div style="font-size:12px;color:#94a3b8;font-family:ui-monospace,monospace;">{_excerpt}...</div>
            </div>
            """
        )

    try:
        _sql_rows = duck_store.run_custom_sql(sql_playground_input.value)
        _sql_status = f"✅ Returned {len(_sql_rows)} row(s)"
    except Exception as _exc:
        _sql_rows = [{"error": str(_exc)}]
        _sql_status = "⚠️ SQL Error"

    mo.md(
        f"""
        ## 2. Graduating from Python Lists to DuckDB (`FLOAT[{embedder.dim}]`)

        All **{indexed_count} chunks** across the 3 Markdown guides in `data/sample_docs/` are now
        embedded and stored in a **DuckDB** table (`rag_chunks`) with fixed-size `FLOAT[{embedder.dim}]` vectors:

        ```sql
        CREATE TABLE rag_chunks (
            chunk_id INTEGER PRIMARY KEY,
            doc_source VARCHAR,
            section_title VARCHAR,
            chunk_text VARCHAR,
            char_length INTEGER,
            embedding FLOAT[{embedder.dim}]
        );
        ```

        ### Live Vector Search in DuckDB SQL (`array_cosine_similarity`)
        {duck_query_input}
        {duck_top_k}

        <div style="background:#0f172a;padding:12px 14px;border-radius:12px;border:1px solid #334155;margin:8px 0;">
          <div style="font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:8px;">
            📊 DuckDB Cosine Similarity Ranking (Top-{duck_top_k.value} Results)
          </div>
          {"".join(_bars_html)}
        </div>

        ---
        ### Interactive DuckDB SQL Playground
        Pick a pre-built analytical recipe or write your own SQL query against `rag_chunks`:
        {sql_recipe_picker}
        {sql_playground_input}
        **Status**: {_sql_status}
        {mo.ui.table(_sql_rows, selection=None)}
        """
    )
    return


if __name__ == "__main__":
    app.run()
