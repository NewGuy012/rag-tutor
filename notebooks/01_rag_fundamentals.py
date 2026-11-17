import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="RAG Tutor 01: Fundamentals & Scratch RAG")


@app.cell
def _():
    import math
    import sys
    from pathlib import Path
    import marimo as mo

    _repo_root = Path(__file__).resolve().parents[1]
    _src_dir = _repo_root / "src"
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    from rag_tutor import (
        SimpleVectorDB,
        build_rag_prompt,
        cosine_similarity,
        dot_product,
        get_default_client,
        get_default_embedder,
        load_ev_facts,
        render_2d_compass_svg,
        render_embedding_barcode_svg,
        render_embedding_space_2d_svg,
        vector_norm,
    )

    llm_client = get_default_client()
    embedder = get_default_embedder()
    ev_facts = load_ev_facts()
    return (
        SimpleVectorDB,
        build_rag_prompt,
        cosine_similarity,
        dot_product,
        embedder,
        ev_facts,
        llm_client,
        math,
        mo,
        render_2d_compass_svg,
        render_embedding_barcode_svg,
        render_embedding_space_2d_svg,
        vector_norm,
    )


@app.cell
def _(llm_client, mo):
    is_online, discovered_models, status_msg = llm_client.check_health()
    _model_options = [m.id for m in discovered_models] if discovered_models else ["educational-fallback"]
    model_dropdown = mo.ui.dropdown(
        options=_model_options,
        value=_model_options[0],
        label="Llama.app Model (`/v1/models`):",
    )
    thinking_toggle = mo.ui.switch(
        value=False,
        label="Enable Model Thinking",
    )
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
    _overview_mermaid = mo.mermaid(
        """
        flowchart LR
            Q["1. User Question"] --> Emb["Embed into 128-D Vector"]
            Emb --> Sim["2. Cosine Similarity vs. ev-facts.txt"]
            Sim --> TopN["Top-N Matched Facts"]
            TopN --> Prompt["3. Augment System Prompt"]
            Prompt --> LLM["Local Llama.app (:9931)"]
            LLM --> Ans["Grounded Cited Answer"]
        """
    )
    mo.md(
        f"""
        # 🦉 RAG Tutor — Notebook 1: The Basic Hows & Whys
        ### *Building & Visualizing a Retrieval-Augmented Generation System from Scratch*

        Welcome to **Notebook 1** of the interactive RAG curriculum! Inspired by Xuan-Son Nguyen's
        [*Code a simple RAG from scratch*](https://huggingface.co/blog/ngxson/make-your-own-rag) and interactive
        explorers from the [**Marimo Gallery**](https://marimo.io/gallery), this notebook demystifies how RAG works
        using **pure Python** and an Electric Vehicle knowledge base (`data/ev-facts.txt`)—before we touch any
        external vector databases or advanced production pipelines.

        {_overview_mermaid}

        ---
        **Local LLM Backend (`http://localhost:9931/v1`)**: {_badge} — _{status_msg}_

        {mo.hstack([model_dropdown, thinking_toggle], justify="start", gap=2)}
        """
    )
    return


@app.cell
def _(mo):
    hallucination_query_picker = mo.ui.dropdown(
        options={
            "EV History: Which 1899 electric car first broke 100 km/h?": (
                "Which electric car was the first road vehicle in history to exceed 100 km/h in April 1899, and who drove it?",
                "In April 1899, a Belgian torpedo-shaped electric car named 'La Jamais Contente' ('The Never Satisfied'), driven by Camille Jenatzy, became the first road vehicle in history to exceed 100 km/h (62 mph), reaching 105.88 km/h.",
            ),
            "Private Protocol: What causes DuckDB error code ERR-VEC-768?": (
                "What causes DuckDB diagnostic error code ERR-VEC-768 and how do I fix it?",
                "Diagnostic ERR-VEC-768 (Dimension Mismatch) occurs when you index documents using a 768-dimension embedding model (like bge-base-en-v1.5) but query the table with a 384-dimension model. Enforce a single embedding dimension across indexing and retrieval.",
            ),
            "EV Engineering: How many kWh equal 1 gallon of gasoline in EPA MPGe?": (
                "How many kilowatt-hours (kWh) of electricity equal 1 gallon of gasoline in the EPA's MPGe rating, and what MPGe does a 4.0 miles/kWh EV achieve?",
                "The EPA measures electric vehicle efficiency using MPGe (Miles Per Gallon equivalent), where 1 gallon of gasoline is defined as chemically equivalent to 33.7 kilowatt-hours (kWh) of electricity. An EV that achieves 4.0 miles per kWh earns an official rating of roughly 135 MPGe.",
            ),
        },
        value="Private Protocol: What causes DuckDB error code ERR-VEC-768?",
        label="Choose a test question:",
    )
    return (hallucination_query_picker,)


@app.cell
def _(
    hallucination_query_picker,
    llm_client,
    mo,
    model_dropdown,
    thinking_toggle,
):
    _q_text, _fact_context = hallucination_query_picker.value

    _no_rag_res = llm_client.generate(
        system_prompt="You are a helpful assistant. Answer the user's question concisely.",
        user_prompt=_q_text,
        model=model_dropdown.value,
        enable_thinking=thinking_toggle.value,
        retrieved_context=[],
    )

    _with_rag_res = llm_client.generate(
        system_prompt=(
            "You are a helpful assistant. Answer the user's question using ONLY "
            f"the following verified context:\n - {_fact_context}"
        ),
        user_prompt=_q_text,
        model=model_dropdown.value,
        enable_thinking=thinking_toggle.value,
        retrieved_context=[_fact_context],
    )

    _no_rag_card = mo.callout(
        mo.md(
            f"**❌ Without RAG (Parametric Memory Only)**\n\n"
            f"*Context Injected:* `None`\n\n"
            f"> {_no_rag_res.content}"
        ),
        kind="warn",
    )
    _with_rag_card = mo.callout(
        mo.md(
            f"**✅ With RAG (Grounded Context Injected)**\n\n"
            f"*Verified Fact:* `{_fact_context}`\n\n"
            f"> {_with_rag_res.content}"
        ),
        kind="success",
    )

    mo.md(
        f"""
        ## 1. Why Do We Need RAG? (The Hallucination & Cutoff Problem)

        Large Language Models (LLMs) store general world knowledge inside their weights (**parametric memory**),
        but they have three fundamental blind spots:
        1. **Private / Internal Data**: They have never seen your team's private documents or custom error codes.
        2. **Knowledge Cutoffs**: They do not know facts created after their training finished.
        3. **Confident Hallucinations**: Without grounded context, an LLM will often guess or fabricate details.

        Select a test question below to compare **Ungrounded Generation (No RAG)** vs. **Grounded Generation (With RAG)**:

        {hallucination_query_picker}

        {mo.hstack([_no_rag_card, _with_rag_card], widths="equal", gap=2)}
        """
    )
    return


@app.cell
def _(mo):
    vec_a_x = mo.ui.slider(-1.0, 1.0, step=0.05, value=0.85, label="Vector A (x)")
    vec_a_y = mo.ui.slider(-1.0, 1.0, step=0.05, value=0.50, label="Vector A (y)")
    vec_b_x = mo.ui.slider(-1.0, 1.0, step=0.05, value=0.75, label="Vector B (x)")
    vec_b_y = mo.ui.slider(-1.0, 1.0, step=0.05, value=0.65, label="Vector B (y)")
    return vec_a_x, vec_a_y, vec_b_x, vec_b_y


@app.cell
def _(
    cosine_similarity,
    dot_product,
    math,
    mo,
    render_2d_compass_svg,
    vec_a_x,
    vec_a_y,
    vec_b_x,
    vec_b_y,
    vector_norm,
):
    _a = [vec_a_x.value, vec_a_y.value]
    _b = [vec_b_x.value, vec_b_y.value]
    _dot = dot_product(_a, _b)
    _norm_a = vector_norm(_a)
    _norm_b = vector_norm(_b)
    _cos = cosine_similarity(_a, _b)
    _clamped = max(-1.0, min(1.0, _cos))
    _angle_deg = math.degrees(math.acos(_clamped)) if (_norm_a > 0 and _norm_b > 0) else 0.0

    _compass_html = render_2d_compass_svg(_a, _b, _dot, _norm_a, _norm_b, _cos, _angle_deg)

    _stat_cards = mo.hstack(
        [
            mo.stat(
                value=f"{_cos:+.4f}",
                label="Cosine Similarity",
                caption=f"Angle θ = {_angle_deg:.1f}°",
                bordered=True,
            ),
            mo.stat(
                value=f"{_dot:+.4f}",
                label="Dot Product (A · B)",
                caption=f"({vec_a_x.value:.2f}×{vec_b_x.value:.2f}) + ({vec_a_y.value:.2f}×{vec_b_y.value:.2f})",
                bordered=True,
            ),
            mo.stat(
                value=f"{_norm_a:.3f} / {_norm_b:.3f}",
                label="Magnitudes ‖A‖ / ‖B‖",
                caption="Euclidean L2 lengths",
                bordered=True,
            ),
        ],
        widths="equal",
        gap=2,
    )

    mo.md(
        f"""
        ## 2. How Retrieval Works: Interactive 2D Vector Compass & Metric Gauges

        To find which facts in our knowledge base match a user's question, we convert text into
        **Embedding Vectors** (lists of floating-point numbers representing meaning) and measure the angle
        between them using **Cosine Similarity**:

        $$\\text{{cosine\\_similarity}}(A, B) = \\frac{{A \\cdot B}}{{\\|A\\| \\cdot \\|B\\|}} = \\frac{{\\sum_{{i=1}}^{{n}} A_i B_i}}{{\\sqrt{{\\sum_{{i=1}}^{{n}} A_i^2}} \\sqrt{{\\sum_{{i=1}}^{{n}} B_i^2}}}}$$

        Drag the sliders below to move 2D vectors **A** (teal) and **B** (amber), inspect the **orthogonal projection**
        and **angle arc $\\theta$**, and compare **Cosine Similarity** against **Dot Product** and **Euclidean Distance**:

        {mo.hstack([vec_a_x, vec_a_y, vec_b_x, vec_b_y], justify="start", gap=2)}

        {_stat_cards}

        {mo.Html(_compass_html)}
        """
    )
    return


@app.cell
def _(SimpleVectorDB, embedder, ev_facts):
    # Build the 50-line Scratch RAG Vector DB in pure Python!
    scratch_db = SimpleVectorDB()
    scratch_db.index_chunks(ev_facts, embedder.embed)
    ev_embeddings = [vec for _, vec in scratch_db.entries]
    return ev_embeddings, scratch_db


@app.cell
def _(mo):
    user_query_input = mo.ui.text(
        value="How does regenerative braking recover kinetic energy and extend brake pad life?",
        label="Ask a question about electric vehicles (updates the 2D Embedding Map & Barcode live):",
        full_width=True,
    )
    top_n_slider = mo.ui.slider(
        start=1,
        stop=6,
        step=1,
        value=3,
        label="Top N Chunks (`top_n`):",
    )
    inspect_rank_dropdown = mo.ui.dropdown(
        options={"Rank #1 (Best Match)": 0, "Rank #2": 1, "Rank #3": 2},
        value="Rank #1 (Best Match)",
        label="Inspect 128-D Barcode for:",
    )
    return inspect_rank_dropdown, top_n_slider, user_query_input


@app.cell
def _(
    build_rag_prompt,
    embedder,
    ev_embeddings,
    ev_facts,
    inspect_rank_dropdown,
    llm_client,
    mo,
    model_dropdown,
    render_embedding_barcode_svg,
    render_embedding_space_2d_svg,
    scratch_db,
    thinking_toggle,
    top_n_slider,
    user_query_input,
):
    _q_vec = embedder.embed(user_query_input.value)
    _retrieved = scratch_db.retrieve(_q_vec, top_n=top_n_slider.value)
    _sys_prompt, _usr_prompt = build_rag_prompt(user_query_input.value, _retrieved)

    _gen = llm_client.generate(
        system_prompt=_sys_prompt,
        user_prompt=_usr_prompt,
        model=model_dropdown.value,
        enable_thinking=thinking_toggle.value,
        retrieved_context=[chunk for chunk, _ in _retrieved],
    )

    # 1. Render 2D Embedding Space Map (PCA projection of all 16 EV facts + live Query Star)
    _space_map_html = render_embedding_space_2d_svg(
        chunks=ev_facts,
        embeddings=ev_embeddings,
        query_text=user_query_input.value,
        query_vec=_q_vec,
        top_n=top_n_slider.value,
    )

    # 2. Render 128-D Embedding Barcode for the chosen rank
    _sel_idx = min(inspect_rank_dropdown.value, len(_retrieved) - 1)
    _sel_chunk, _sel_score = _retrieved[_sel_idx]
    _sel_vec = embedder.embed(_sel_chunk)
    _barcode_html = render_embedding_barcode_svg(
        query_vec=_q_vec,
        doc_vec=_sel_vec,
        cosine_score=_sel_score,
    )

    _rows_md = "\n".join(
        f"| **#{idx + 1}** | `{score:.4f}` | {chunk} |"
        for idx, (chunk, score) in enumerate(_retrieved)
    )

    _tabs = mo.ui.tabs(
        {
            "🌌 2D Embedding Space & 128-D Barcode": mo.vstack(
                [
                    mo.Html(_space_map_html),
                    mo.hstack([inspect_rank_dropdown], justify="start"),
                    mo.Html(_barcode_html),
                ]
            ),
            "📋 Top-N Retrieved Chunks Table": mo.md(
                f"""
                | Rank | Cosine Similarity | Retrieved Fact from `ev-facts.txt` |
                | :--- | :--- | :--- |
                {_rows_md}
                """
            ),
            "🔍 Augmented System Prompt": mo.md(
                f"""
                ```text
                SYSTEM PROMPT:
                {_sys_prompt}

                USER PROMPT:
                {_usr_prompt}
                ```
                """
            ),
        }
    )

    mo.md(
        f"""
        ## 3. Your First 50-Line RAG from Scratch (`ev-facts.txt`)

        Following [Xuan-Son Nguyen's Hugging Face guide](https://huggingface.co/blog/ngxson/make-your-own-rag),
        we have indexed all **{len(scratch_db.entries)} facts** from `data/ev-facts.txt` into a plain Python list:
        `VECTOR_DB = [(chunk_text, embedding_vector), ...]`.

        Type any question about electric vehicles below (e.g., *"What is the difference between LFP and NMC batteries?"*,
        *"Why do 800-volt EV architectures charge faster?"*, *"How do automotive heat pumps help in cold winter weather?"*,
        *"Why does DC fast charging slow down above 80% state of charge?"*) and watch the **2D Embedding Space Map**
        and **128-D Barcode** react live:

        {user_query_input}
        {mo.hstack([top_n_slider, inspect_rank_dropdown], justify="start", gap=3)}

        {_tabs}

        ### Grounded LLM Response (`{_gen.model}` · `{_gen.latency_ms:.0f} ms`)
        > {_gen.content}

        ---
        **Next Up**: In `notebooks/02_chunking_and_duckdb.py`, we'll visually explore document chunking & sliding overlap on a **Document Timeline Minimap** and index our vectors into **DuckDB**!
        """
    )
    return


if __name__ == "__main__":
    app.run()
