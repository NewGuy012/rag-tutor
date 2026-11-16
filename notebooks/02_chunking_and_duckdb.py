import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="RAG Tutor 02: Chunking & DuckDB Vectors")


@app.cell
def _():
    import sys
    from pathlib import Path
    import marimo as mo

    _repo_root = Path(__file__).resolve().parents[1]
    _src_dir = _repo_root / "src"
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    from rag_tutor.chunker import (
        chunk_fixed_character,
        chunk_markdown_sections,
        load_sample_markdown_corpus,
    )
    from rag_tutor.duckdb_store import DuckDBVectorStore
    from rag_tutor.llm_backend import get_default_embedder

    sample_docs_dir = _repo_root / "data" / "sample_docs"
    doc_files = {p.name: p.read_text(encoding="utf-8") for p in sorted(sample_docs_dir.glob("*.md"))}
    embedder = get_default_embedder()
    return (
        DuckDBVectorStore,
        Path,
        chunk_fixed_character,
        chunk_markdown_sections,
        doc_files,
        embedder,
        load_sample_markdown_corpus,
        mo,
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
    chunk_fixed_character,
    chunk_markdown_sections,
    chunk_overlap_slider,
    chunk_size_slider,
    doc_files,
    doc_selector,
    mo,
    strategy_selector,
):
    _raw_text = doc_files[doc_selector.value]
    if strategy_selector.value == "fixed":
        preview_chunks = chunk_fixed_character(
            _raw_text,
            chunk_size=chunk_size_slider.value,
            chunk_overlap=chunk_overlap_slider.value,
            doc_source=doc_selector.value,
        )
    else:
        preview_chunks = chunk_markdown_sections(
            _raw_text,
            max_chunk_size=chunk_size_slider.value,
            chunk_overlap=chunk_overlap_slider.value,
            doc_source=doc_selector.value,
        )

    _palette = ["#1e293b", "#172554", "#134e4a", "#3b0764", "#3f2e1e"]
    _borders = ["#38bdf8", "#60a5fa", "#2dd4bf", "#c084fc", "#f59e0b"]
    _cards = []
    for _idx, _ch in enumerate(preview_chunks):
        _bg = _palette[_idx % len(_palette)]
        _bd = _borders[_idx % len(_borders)]
        _escaped = (
            _ch.text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        _cards.append(
            f"""
            <div style="background:{_bg};border-left:4px solid {_bd};padding:10px 14px;margin-bottom:8px;border-radius:6px;color:#f1f5f9;font-size:13px;">
              <div style="font-weight:700;color:{_bd};margin-bottom:4px;">
                Chunk #{_ch.chunk_id} · Section: {_ch.section_title}
                <span style="float:right;font-weight:400;color:#cbd5e1;">
                  {_ch.char_length} chars ({_ch.word_count} words) · overlap={_ch.overlap_chars} chars
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
        ## 1. Interactive Document Chunking Lab

        Adjust the controls below to see how **Chunk Strategy**, **Max Chunk Size**, and **Sliding Overlap**
        split `{doc_selector.value}` into **{len(preview_chunks)} chunks**:

        {mo.hstack([doc_selector, strategy_selector], justify="start", gap=2)}
        {mo.hstack([chunk_size_slider, chunk_overlap_slider], justify="start", gap=2)}

        {mo.Html("".join(_cards))}
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
    sql_playground_input = mo.ui.text_area(
        value=(
            "SELECT chunk_id, doc_source, section_title, char_length\n"
            "FROM rag_chunks\n"
            "ORDER BY char_length DESC\n"
            "LIMIT 6;"
        ),
        label="Interactive DuckDB SQL Playground (query `rag_chunks` directly):",
        rows=4,
        full_width=True,
    )
    return duck_query_input, duck_top_k, sql_playground_input


@app.cell
def _(
    duck_query_input,
    duck_store,
    duck_top_k,
    embedder,
    indexed_count,
    mo,
    sql_playground_input,
):
    _vec_hits = duck_store.search_vector(duck_query_input.value, top_k=duck_top_k.value)
    _hit_rows = "\n".join(
        f"| **#{idx + 1}** | `{r.score:.4f}` | `{r.doc_source}` | **{r.section_title}** | {r.text[:160].replace(chr(10), ' ')}... |"
        for idx, r in enumerate(_vec_hits)
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
        embedded and stored in a **DuckDB** table (`rag_chunks`) with schema:

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

        | Rank | DuckDB `vector_score` | Document | Section | Chunk Excerpt |
        | :--- | :--- | :--- | :--- | :--- |
        {_hit_rows}

        ---
        ### Try Your Own SQL Queries Against `rag_chunks`
        {sql_playground_input}
        **Status**: {_sql_status}
        """
    )
    return


@app.cell
def _(duck_store, mo, sql_playground_input):
    try:
        _table_data = duck_store.run_custom_sql(sql_playground_input.value)
    except Exception as _e:
        _table_data = [{"sql_error": str(_e)}]
    mo.ui.table(_table_data)
    return


if __name__ == "__main__":
    app.run()
