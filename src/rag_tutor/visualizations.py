"""Interactive SVG, Heatmap, Embedding Projection, and Mermaid Visualization Builders.

Inspired by the Marimo Gallery (`marimo.io/gallery`, `marimo-team/gallery-examples`,
`marimo-team/learn`) and Explorable Explanations, this module provides pure-Python,
zero-dependency visual components that turn abstract RAG concepts (128-D embeddings,
2D semantic space projections, chunk overlap boundaries, RRF fusion curves,
Cross-Encoder rank slopegraphs, and Agentic execution traces) into reactive visuals.
"""

from __future__ import annotations

import html
import math
import re
from typing import Sequence

from .agentic_rag import AgenticTrace
from .chunker import Chunk, extract_markdown_headings
from .hybrid_search import HybridSearchResult
from .llm_backend import SEMANTIC_CLUSTERS, split_sentences, tokenize
from .reranker import RerankedCandidate
from .scratch_rag import cosine_similarity


def _esc(text: str) -> str:
    """HTML-escape text for safe inclusion in SVG/HTML attributes and nodes."""
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# 1. Notebook 1: Enhanced 2D Cosine Compass + Metric Comparison Gauges
# ---------------------------------------------------------------------------


def render_2d_compass_svg(
    vec_a: Sequence[float],
    vec_b: Sequence[float],
    dot_val: float,
    norm_a: float,
    norm_b: float,
    cos_val: float,
    angle_deg: float,
) -> str:
    """Render an interactive 2D vector compass with angle arc, projection line,
    and side-by-side Cosine vs. Dot Product vs. Euclidean similarity gauges."""
    cx, cy, scale = 150.0, 150.0, 108.0
    ax = cx + vec_a[0] * scale
    ay = cy - vec_a[1] * scale
    bx = cx + vec_b[0] * scale
    by = cy - vec_b[1] * scale

    # Unit vectors for angle arc and orthogonal projection of B onto A
    proj_line_svg = ""
    arc_svg = ""
    if norm_a > 1e-6 and norm_b > 1e-6:
        ua_x, ua_y = vec_a[0] / norm_a, vec_a[1] / norm_a
        proj_len = dot_val / norm_a  # scalar projection of B onto A
        px = cx + (ua_x * proj_len) * scale
        py = cy - (ua_y * proj_len) * scale
        proj_line_svg = (
            f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
            f'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3 3" />'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="#94a3b8" />'
        )

        # Draw angle sector arc at radius 38px
        arc_r = 38.0
        ang_a = math.atan2(-vec_a[1], vec_a[0])
        ang_b = math.atan2(-vec_b[1], vec_b[0])
        start_x = cx + arc_r * math.cos(ang_a)
        start_y = cy + arc_r * math.sin(ang_a)
        end_x = cx + arc_r * math.cos(ang_b)
        end_y = cy + arc_r * math.sin(ang_b)

        # Determine sweep direction for shortest angle
        diff = (ang_b - ang_a + math.pi) % (2 * math.pi) - math.pi
        sweep = 1 if diff >= 0 else 0
        mid_ang = ang_a + diff / 2.0
        lbl_x = cx + (arc_r + 16) * math.cos(mid_ang)
        lbl_y = cy + (arc_r + 16) * math.sin(mid_ang)
        arc_svg = (
            f'<path d="M {start_x:.1f} {start_y:.1f} A {arc_r:.1f} {arc_r:.1f} 0 0 {sweep} {end_x:.1f} {end_y:.1f}" '
            f'fill="none" stroke="#a855f7" stroke-width="2.2" />'
            f'<text x="{lbl_x:.1f}" y="{lbl_y:.1f}" fill="#e9d5ff" font-size="11" '
            f'font-family="monospace" text-anchor="middle" dominant-baseline="middle">{angle_deg:.0f}°</text>'
        )

    euclid_dist = math.sqrt((vec_a[0] - vec_b[0]) ** 2 + (vec_a[1] - vec_b[1]) ** 2)
    euclid_sim = 1.0 / (1.0 + euclid_dist)

    # Normalize bar widths for visual comparison
    cos_pct = max(0.0, min(100.0, (cos_val + 1.0) * 50.0))
    dot_clamped = max(-2.0, min(2.0, dot_val))
    dot_pct = max(0.0, min(100.0, (dot_clamped + 2.0) * 25.0))
    euc_pct = max(0.0, min(100.0, euclid_sim * 100.0))

    if cos_val >= 0.75:
        verdict = "🟢 High Semantic Alignment (Same Direction)"
        verdict_color = "#2dd4bf"
    elif cos_val >= 0.25:
        verdict = "🟡 Moderate Similarity (Acute Angle)"
        verdict_color = "#fbbf24"
    elif cos_val >= -0.25:
        verdict = "⚪ Near-Orthogonal (~90° Unrelated)"
        verdict_color = "#94a3b8"
    else:
        verdict = "🔴 Opposite Semantic Direction (< 0)"
        verdict_color = "#f87171"

    return f"""
    <div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;background:#0f172a;padding:14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;">
      <svg width="300" height="300" viewBox="0 0 300 300" style="background:#020617;border-radius:10px;border:1px solid #1e293b;flex-shrink:0;">
        <!-- Reference circles r=0.5 and r=1.0 -->
        <circle cx="{cx}" cy="{cy}" r="{scale * 0.5:.1f}" fill="none" stroke="#1e293b" stroke-dasharray="3 3"/>
        <circle cx="{cx}" cy="{cy}" r="{scale:.1f}" fill="none" stroke="#334155" stroke-dasharray="4 4"/>
        <!-- Axes -->
        <line x1="16" y1="{cy}" x2="284" y2="{cy}" stroke="#334155" stroke-width="1"/>
        <line x1="{cx}" y1="16" x2="{cx}" y2="284" stroke="#334155" stroke-width="1"/>
        <text x="278" y="{cy - 6}" fill="#64748b" font-size="10" font-family="monospace" text-anchor="end">+x</text>
        <text x="{cx + 6}" y="26" fill="#64748b" font-size="10" font-family="monospace">+y</text>
        <!-- Projection line & Angle Arc -->
        {proj_line_svg}
        {arc_svg}
        <!-- Vector A (Teal) -->
        <line x1="{cx}" y1="{cy}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="#2dd4bf" stroke-width="4" stroke-linecap="round"/>
        <circle cx="{ax:.1f}" cy="{ay:.1f}" r="6" fill="#2dd4bf"/>
        <text x="{ax + 8:.1f}" y="{ay - 8:.1f}" fill="#2dd4bf" font-size="12" font-weight="bold" font-family="monospace">A ({vec_a[0]:.2f}, {vec_a[1]:.2f})</text>
        <!-- Vector B (Amber) -->
        <line x1="{cx}" y1="{cy}" x2="{bx:.1f}" y2="{by:.1f}" stroke="#f59e0b" stroke-width="4" stroke-linecap="round"/>
        <circle cx="{bx:.1f}" cy="{by:.1f}" r="6" fill="#f59e0b"/>
        <text x="{bx + 8:.1f}" y="{by + 15:.1f}" fill="#f59e0b" font-size="12" font-weight="bold" font-family="monospace">B ({vec_b[0]:.2f}, {vec_b[1]:.2f})</text>
      </svg>

      <div style="flex:1;min-width:260px;font-family:system-ui,-apple-system,sans-serif;">
        <div style="font-size:14px;font-weight:700;color:{verdict_color};margin-bottom:10px;">
          {verdict}
        </div>
        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
            <span style="color:#cbd5e1;"><strong>Cosine Similarity</strong> (Angle-only, scale-invariant)</span>
            <span style="font-family:monospace;color:#2dd4bf;font-weight:700;">{cos_val:+.4f} (θ = {angle_deg:.1f}°)</span>
          </div>
          <div style="background:#1e293b;height:10px;border-radius:5px;overflow:hidden;position:relative;">
            <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#475569;"></div>
            <div style="width:{cos_pct:.1f}%;background:linear-gradient(90deg,#f43f5e,#eab308,#2dd4bf);height:100%;border-radius:5px;"></div>
          </div>
        </div>

        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
            <span style="color:#cbd5e1;"><strong>Raw Dot Product (A · B)</strong> (Sensitive to vector length)</span>
            <span style="font-family:monospace;color:#f59e0b;font-weight:700;">{dot_val:+.4f}</span>
          </div>
          <div style="background:#1e293b;height:10px;border-radius:5px;overflow:hidden;">
            <div style="width:{dot_pct:.1f}%;background:#f59e0b;height:100%;border-radius:5px;"></div>
          </div>
        </div>

        <div style="margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px;">
            <span style="color:#cbd5e1;"><strong>Euclidean Distance ‖A − B‖</strong> (Tip-to-tip distance)</span>
            <span style="font-family:monospace;color:#c084fc;font-weight:700;">dist = {euclid_dist:.3f} (sim = {euclid_sim:.3f})</span>
          </div>
          <div style="background:#1e293b;height:10px;border-radius:5px;overflow:hidden;">
            <div style="width:{euc_pct:.1f}%;background:#c084fc;height:100%;border-radius:5px;"></div>
          </div>
        </div>

        <div style="font-size:12px;color:#94a3b8;background:#020617;padding:8px 10px;border-radius:6px;border-left:3px solid #a855f7;">
          💡 <strong>Visual Intuition:</strong> The dashed gray line shows the orthogonal projection of <strong>B</strong> onto <strong>A</strong>.
          Try scaling <strong>A</strong> from <code>(0.9, 0.0)</code> down to <code>(0.2, 0.0)</code>:
          <em>Dot Product</em> and <em>Euclidean Distance</em> change, but <strong>Cosine Similarity stays constant!</strong>
        </div>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# 2. Notebook 1: 128-D Embedding Barcode & Element-Wise Contribution Heatmap
# ---------------------------------------------------------------------------


def render_embedding_barcode_svg(
    query_vec: Sequence[float],
    doc_vec: Sequence[float],
    cosine_score: float,
) -> str:
    """Render a 3-row 128-D embedding barcode comparing Query, Chunk, and Q_i * D_i."""
    dim = min(len(query_vec), len(doc_vec))
    if dim == 0:
        return ""

    cell_w = 5.2
    width = int(dim * cell_w + 135)
    x_offset = 125

    def val_to_color(val: float, is_product: bool = False) -> str:
        t = min(1.0, abs(val) * (18.0 if is_product else 4.2))
        if is_product:
            tr, tg, tb = (45, 212, 191) if val >= 0 else (244, 63, 94)
        else:
            tr, tg, tb = (56, 189, 248) if val >= 0 else (251, 146, 60)
        return f"rgb({int(15 + (tr - 15) * t)},{int(23 + (tg - 23) * t)},{int(42 + (tb - 42) * t)})"

    q_rects: list[str] = []
    d_rects: list[str] = []
    p_rects: list[str] = []
    contributions: list[tuple[float, int, str]] = []

    num_clusters = len(SEMANTIC_CLUSTERS)
    for i in range(dim):
        qv = query_vec[i]
        dv = doc_vec[i]
        prod = qv * dv
        x = x_offset + i * cell_w

        if i < num_clusters:
            dim_name = f"Dim {i} (Semantic: {SEMANTIC_CLUSTERS[i][0]})"
        elif i < int(dim * 0.6):
            dim_name = f"Dim {i} (Lexical Token Hash)"
        else:
            dim_name = f"Dim {i} (Char 3-Gram Subword)"

        contributions.append((prod, i, dim_name))

        q_rects.append(
            f'<rect x="{x:.1f}" y="28" width="{cell_w - 0.6:.1f}" height="20" '
            f'fill="{val_to_color(qv)}"><title>{_esc(dim_name)}: Q[{i}] = {qv:+.4f}</title></rect>'
        )
        d_rects.append(
            f'<rect x="{x:.1f}" y="56" width="{cell_w - 0.6:.1f}" height="20" '
            f'fill="{val_to_color(dv)}"><title>{_esc(dim_name)}: D[{i}] = {dv:+.4f}</title></rect>'
        )
        p_rects.append(
            f'<rect x="{x:.1f}" y="84" width="{cell_w - 0.6:.1f}" height="20" '
            f'fill="{val_to_color(prod, is_product=True)}"><title>{_esc(dim_name)}: Q[{i}]*D[{i}] = {prod:+.4f}</title></rect>'
        )

    contributions.sort(key=lambda item: item[0], reverse=True)
    top_dims_badges = " ".join(
        f'<span style="background:#134e4a;color:#5eead4;padding:2px 8px;border-radius:999px;font-size:11px;font-family:monospace;">'
        f'{_esc(lbl)} (+{sc:.3f})</span>'
        for sc, _, lbl in contributions[:4]
        if sc > 0.001
    ) or '<span style="color:#94a3b8;font-size:11px;">Low dimensional overlap</span>'

    sem_end_x = x_offset + num_clusters * cell_w
    lex_end_x = x_offset + int(dim * 0.6) * cell_w
    total_end_x = x_offset + dim * cell_w

    return f"""
    <div style="background:#0f172a;padding:12px 14px;border-radius:10px;border:1px solid #334155;color:#f8fafc;margin-top:8px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
        <span style="font-size:13px;font-weight:700;color:#e2e8f0;">
          🔬 128-D Embedding Barcode Inspector (Hover any bar to inspect dimension)
        </span>
        <span style="font-family:monospace;font-size:12px;color:#2dd4bf;">
          ∑ (Qᵢ × Dᵢ) = Cosine Similarity = <strong>{cosine_score:.4f}</strong>
        </span>
      </div>
      <div style="overflow-x:auto;">
        <svg width="{width}" height="122" viewBox="0 0 {width} 122" style="background:#020617;border-radius:8px;border:1px solid #1e293b;">
          <!-- Region Headers -->
          <text x="{(x_offset + sem_end_x) / 2:.1f}" y="16" fill="#38bdf8" font-size="10" font-family="monospace" text-anchor="middle">0..14 Semantic</text>
          <line x1="{sem_end_x:.1f}" y1="6" x2="{sem_end_x:.1f}" y2="112" stroke="#334155" stroke-dasharray="2 2"/>
          <text x="{(sem_end_x + lex_end_x) / 2:.1f}" y="16" fill="#a78bfa" font-size="10" font-family="monospace" text-anchor="middle">15..76 Lexical Hash Projections</text>
          <line x1="{lex_end_x:.1f}" y1="6" x2="{lex_end_x:.1f}" y2="112" stroke="#334155" stroke-dasharray="2 2"/>
          <text x="{(lex_end_x + total_end_x) / 2:.1f}" y="16" fill="#f59e0b" font-size="10" font-family="monospace" text-anchor="middle">77..127 Char 3-Grams</text>

          <!-- Row Labels -->
          <text x="10" y="42" fill="#38bdf8" font-size="11" font-family="monospace" font-weight="bold">Query (128-D)</text>
          <text x="10" y="70" fill="#cbd5e1" font-size="11" font-family="monospace" font-weight="bold">Chunk (128-D)</text>
          <text x="10" y="98" fill="#2dd4bf" font-size="11" font-family="monospace" font-weight="bold">Product Qᵢ×Dᵢ</text>

          {"".join(q_rects)}
          {"".join(d_rects)}
          {"".join(p_rects)}
        </svg>
      </div>
      <div style="margin-top:8px;font-size:12px;color:#cbd5e1;display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
        <span><strong>Top Aligned Dimensions:</strong></span> {top_dims_badges}
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# 3. Notebook 1: 2D Embedding Space Map (All Corpus Chunks + Live Query Star)
# ---------------------------------------------------------------------------

TOPIC_PALETTE: list[tuple[str, str, set[str]]] = [
    ("Battery & Chemistry", "#c084fc", {"battery", "lfp", "nmc", "solid-state", "electrolyte", "density"}),
    ("Charging & 800V/V2H", "#38bdf8", {"charging", "800-volt", "nacs", "preconditioning", "bidirectional", "v2h", "soc"}),
    ("Motors & Regen Braking", "#2dd4bf", {"regenerative", "braking", "motor", "torque", "pmsm", "induction", "pedal"}),
    ("Efficiency & Winter Thermal", "#fbbf24", {"mpge", "efficiency", "aerodynamic", "heat", "pump", "winter", "cold"}),
    ("History, Safety & BMS", "#fb7185", {"1899", "jamais", "apollo", "lunar", "bms", "pyrotechnic", "12-volt"}),
]


def _classify_ev_topic(text: str) -> tuple[str, str]:
    tokens = set(tokenize(text))
    best_label = "History, Safety & BMS"
    best_color = "#fb7185"
    best_hits = -1
    for label, color, kw in TOPIC_PALETTE:
        hits = len(tokens & kw)
        if hits > best_hits:
            best_hits = hits
            best_label = label
            best_color = color
    return best_label, best_color


def _project_2d_deterministic(
    vectors: list[list[float]],
    query_vec: list[float],
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Project high-dimensional embeddings into 2D using pure-Python centered
    orthogonal power iteration (equivalent to top-2 Principal Components)."""
    all_vecs = list(vectors)
    n = len(all_vecs)
    if n == 0:
        return [], (0.0, 0.0)
    dim = len(all_vecs[0])

    # Compute mean vector of corpus
    mean_vec = [sum(v[d] for v in all_vecs) / n for d in range(dim)]
    centered = [[v[d] - mean_vec[d] for d in range(dim)] for v in all_vecs]
    q_centered = [query_vec[d] - mean_vec[d] for d in range(dim)]

    def power_axis(deflate_axis: list[float] | None = None) -> list[float]:
        # Deterministic seed vector
        w = [math.sin((d + 1) * 1.7) if deflate_axis is None else math.cos((d + 1) * 2.3) for d in range(dim)]
        for _ in range(18):
            # Compute X^T (X w)
            new_w = [0.0] * dim
            for row in centered:
                dot = sum(row[d] * w[d] for d in range(dim))
                for d in range(dim):
                    new_w[d] += dot * row[d]
            if deflate_axis is not None:
                proj = sum(new_w[d] * deflate_axis[d] for d in range(dim))
                new_w = [new_w[d] - proj * deflate_axis[d] for d in range(dim)]
            norm = math.sqrt(sum(x * x for x in new_w)) or 1.0
            w = [x / norm for x in new_w]
        return w

    pc1 = power_axis(None)
    pc2 = power_axis(pc1)

    pts = [
        (
            sum(row[d] * pc1[d] for d in range(dim)),
            sum(row[d] * pc2[d] for d in range(dim)),
        )
        for row in centered
    ]
    q_pt = (
        sum(q_centered[d] * pc1[d] for d in range(dim)),
        sum(q_centered[d] * pc2[d] for d in range(dim)),
    )
    return pts, q_pt


def render_embedding_space_2d_svg(
    chunks: list[str],
    embeddings: list[list[float]],
    query_text: str,
    query_vec: list[float],
    top_n: int = 3,
) -> str:
    """Render an interactive 2D PCA projection of all corpus embeddings with the
    live Query point and rays connecting to the Top-N nearest neighbors."""
    pts, q_pt = _project_2d_deterministic(embeddings, query_vec)
    if not pts:
        return ""

    # Compute cosine similarities to identify Top-N indices
    scored = [
        (idx, cosine_similarity(query_vec, emb))
        for idx, emb in enumerate(embeddings)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    top_map = {idx: (rank + 1, sim) for rank, (idx, sim) in enumerate(scored[:top_n])}
    all_sim_map = {idx: sim for idx, sim in scored}

    all_x = [p[0] for p in pts] + [q_pt[0]]
    all_y = [p[1] for p in pts] + [q_pt[1]]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max(0.05, max_x - min_x)
    span_y = max(0.05, max_y - min_y)

    width, height, pad = 640, 310, 42

    def to_screen(pt: tuple[float, float]) -> tuple[float, float]:
        sx = pad + ((pt[0] - min_x) / span_x) * (width - 2 * pad)
        sy = height - pad - ((pt[1] - min_y) / span_y) * (height - 2 * pad)
        return sx, sy

    qx, qy = to_screen(q_pt)

    rays_svg: list[str] = []
    nodes_svg: list[str] = []

    for idx, (pt, chunk_text) in enumerate(zip(pts, chunks)):
        cx, cy = to_screen(pt)
        topic_lbl, color = _classify_ev_topic(chunk_text)
        sim_val = all_sim_map[idx]
        short_lbl = f"Fact #{idx + 1}"

        if idx in top_map:
            rank, _ = top_map[idx]
            # Draw glowing ray from Query to this Top-N chunk
            rays_svg.append(
                f'<line x1="{qx:.1f}" y1="{qy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
                f'stroke="#2dd4bf" stroke-width="{max(1.5, 3.2 - rank * 0.4):.1f}" stroke-dasharray="4 3" opacity="0.85"/>'
            )
            nodes_svg.append(
                f'<g>'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="none" stroke="#2dd4bf" stroke-width="2" />'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{color}" stroke="#020617" stroke-width="1.5">'
                f'<title>Top #{rank} (cos={sim_val:.4f}) · {_esc(topic_lbl)}&#10;{_esc(chunk_text)}</title></circle>'
                f'<text x="{cx + 12:.1f}" y="{cy + 4:.1f}" fill="#f8fafc" font-size="11" font-weight="bold" font-family="monospace">'
                f'#{rank} ({sim_val:.2f})</text>'
                f'</g>'
            )
        else:
            nodes_svg.append(
                f'<g>'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{color}" opacity="0.72" stroke="#0f172a" stroke-width="1">'
                f'<title>{_esc(short_lbl)} (cos={sim_val:.4f}) · {_esc(topic_lbl)}&#10;{_esc(chunk_text)}</title></circle>'
                f'</g>'
            )

    legend_items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:#cbd5e1;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{col};display:inline-block;"></span>'
        f'{_esc(lbl)}</span>'
        for lbl, col, _ in TOPIC_PALETTE
    )

    return f"""
    <div style="background:#0f172a;padding:12px 14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
        <span style="font-size:13px;font-weight:700;color:#e2e8f0;">
          🌌 2D Embedding Space Map (PCA Projection of 128-D Vectors + Live Query Neighborhood)
        </span>
        <span style="font-size:11px;color:#94a3b8;">
          Hover any point to read the fact · Dashed teal beams connect your Query (★) to Top-{top_n} matches
        </span>
      </div>
      <div style="overflow-x:auto;">
        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#020617;border-radius:8px;border:1px solid #1e293b;width:100%;max-width:{width}px;">
          <!-- Subtle grid -->
          <line x1="{width / 2:.1f}" y1="15" x2="{width / 2:.1f}" y2="{height - 15}" stroke="#1e293b" stroke-dasharray="3 3"/>
          <line x1="15" y1="{height / 2:.1f}" x2="{width - 15}" y2="{height / 2:.1f}" stroke="#1e293b" stroke-dasharray="3 3"/>
          <text x="{width - 12}" y="{height - 8}" fill="#475569" font-size="10" font-family="monospace" text-anchor="end">PC1 (Semantic Axis 1)</text>
          <text x="12" y="18" fill="#475569" font-size="10" font-family="monospace">PC2 (Semantic Axis 2)</text>

          {"".join(rays_svg)}
          {"".join(nodes_svg)}

          <!-- Query Star -->
          <circle cx="{qx:.1f}" cy="{qy:.1f}" r="14" fill="#2dd4bf" opacity="0.22"/>
          <circle cx="{qx:.1f}" cy="{qy:.1f}" r="7.5" fill="#14b8a6" stroke="#ffffff" stroke-width="2">
            <title>Live User Query: {_esc(query_text)}</title>
          </circle>
          <text x="{qx:.1f}" y="{qy - 14:.1f}" fill="#5eead4" font-size="12" font-weight="bold" font-family="monospace" text-anchor="middle">★ YOUR QUERY</text>
        </svg>
      </div>
      <div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:8px;padding-top:6px;border-top:1px solid #1e293b;">
        {legend_items}
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# 4. Notebook 2: Document Chunking Timeline Minimap & Length Histogram
# ---------------------------------------------------------------------------


def render_chunk_timeline_svg(
    raw_text: str,
    chunks: list[Chunk],
    strategy_name: str,
    target_chunk_size: int,
) -> str:
    """Render a two-part visual for Notebook 2:
    (1) Document span minimap showing markdown heading boundaries, staggered chunk spans,
        and highlighted amber overlap windows.
    (2) Chunk character length bar chart vs. max_chunk_size threshold."""
    total_chars = max(1, len(raw_text))
    width = 660
    pad_l, pad_r = 75, 25
    track_w = width - pad_l - pad_r

    heading_markers = extract_markdown_headings(raw_text)

    palette = ["#38bdf8", "#60a5fa", "#2dd4bf", "#c084fc", "#f472b6"]
    lane_y = [48, 72, 96]

    heading_svg: list[str] = []
    for h_pos, h_title in heading_markers:
        hx = pad_l + (h_pos / total_chars) * track_w
        heading_svg.append(
            f'<line x1="{hx:.1f}" y1="22" x2="{hx:.1f}" y2="122" stroke="#64748b" stroke-width="1" stroke-dasharray="2 2">'
            f'<title>Heading @ char {h_pos}: {_esc(h_title)}</title></line>'
            f'<circle cx="{hx:.1f}" cy="22" r="3" fill="#94a3b8"><title>Heading: {_esc(h_title)}</title></circle>'
        )

    chunk_bars_svg: list[str] = []
    overlap_zones_svg: list[str] = []

    for idx, ch in enumerate(chunks):
        x1 = pad_l + (max(0, ch.start_char) / total_chars) * track_w
        x2 = pad_l + (min(total_chars, ch.end_char) / total_chars) * track_w
        bw = max(6.0, x2 - x1)
        y = lane_y[idx % len(lane_y)]
        col = palette[idx % len(palette)]

        chunk_bars_svg.append(
            f'<g>'
            f'<rect x="{x1:.1f}" y="{y}" width="{bw:.1f}" height="16" rx="4" fill="{col}" opacity="0.88" stroke="#020617" stroke-width="1">'
            f'<title>Chunk #{ch.chunk_id} ({ch.section_title}) · chars {ch.start_char}..{ch.end_char} ({ch.char_length} chars, overlap={ch.overlap_chars})</title></rect>'
            f'<text x="{x1 + bw / 2:.1f}" y="{y + 11.5:.1f}" fill="#020617" font-size="10" font-weight="bold" font-family="monospace" text-anchor="middle">#{ch.chunk_id}</text>'
            f'</g>'
        )

        # Highlight overlap with previous chunk if intervals intersect
        if idx > 0:
            prev = chunks[idx - 1]
            if ch.start_char < prev.end_char:
                ox1 = pad_l + (ch.start_char / total_chars) * track_w
                ox2 = pad_l + (min(total_chars, prev.end_char) / total_chars) * track_w
                ow = max(3.0, ox2 - ox1)
                overlap_zones_svg.append(
                    f'<rect x="{ox1:.1f}" y="38" width="{ow:.1f}" height="80" fill="#f59e0b" opacity="0.26">'
                    f'<title>Sliding Overlap between Chunk #{prev.chunk_id} and #{ch.chunk_id} ({ch.overlap_chars} chars)</title></rect>'
                )

    # Bottom mini bar chart: Chunk character lengths vs target_chunk_size
    max_len = max([target_chunk_size] + [c.char_length for c in chunks] + [100])
    hist_top = 148
    hist_h = 68
    hist_bottom = hist_top + hist_h
    target_y = hist_bottom - (target_chunk_size / max_len) * hist_h

    hist_bars: list[str] = []
    n_chunks = max(1, len(chunks))
    slot_w = track_w / n_chunks
    bar_w = max(8.0, min(38.0, slot_w * 0.72))

    for idx, ch in enumerate(chunks):
        bx = pad_l + idx * slot_w + (slot_w - bar_w) / 2.0
        bh = max(4.0, (ch.char_length / max_len) * hist_h)
        by = hist_bottom - bh
        col = palette[idx % len(palette)]
        hist_bars.append(
            f'<g>'
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3" fill="{col}">'
            f'<title>Chunk #{ch.chunk_id}: {ch.char_length} chars ({ch.word_count} words)</title></rect>'
            f'<text x="{bx + bar_w / 2:.1f}" y="{by - 4:.1f}" fill="#cbd5e1" font-size="9.5" font-family="monospace" text-anchor="middle">{ch.char_length}</text>'
            f'<text x="{bx + bar_w / 2:.1f}" y="{hist_bottom + 12:.1f}" fill="#94a3b8" font-size="9.5" font-family="monospace" text-anchor="middle">#{ch.chunk_id}</text>'
            f'</g>'
        )

    return f"""
    <div style="background:#0f172a;padding:12px 14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;margin-bottom:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;">
        <span style="font-size:13px;font-weight:700;color:#e2e8f0;">
          📏 Document Chunking Minimap & Overlap Visualizer ({_esc(strategy_name)})
        </span>
        <span style="font-size:11px;color:#cbd5e1;">
          <span style="color:#94a3b8;">● Dashed Vertical Lines = Markdown Headings</span> ·
          <span style="color:#f59e0b;">█ Amber Highlights = Sliding Overlap Zones</span>
        </span>
      </div>
      <div style="overflow-x:auto;">
        <svg width="{width}" height="236" viewBox="0 0 {width} 236" style="background:#020617;border-radius:8px;border:1px solid #1e293b;width:100%;max-width:{width}px;">
          <!-- Track labels -->
          <text x="10" y="25" fill="#94a3b8" font-size="10" font-family="monospace">Headings</text>
          <text x="10" y="60" fill="#cbd5e1" font-size="10" font-family="monospace">Swimlane 1</text>
          <text x="10" y="84" fill="#cbd5e1" font-size="10" font-family="monospace">Swimlane 2</text>
          <text x="10" y="108" fill="#cbd5e1" font-size="10" font-family="monospace">Swimlane 3</text>

          <!-- Swimlane guide lines -->
          <line x1="{pad_l}" y1="56" x2="{width - pad_r}" y2="56" stroke="#1e293b" stroke-width="1"/>
          <line x1="{pad_l}" y1="80" x2="{width - pad_r}" y2="80" stroke="#1e293b" stroke-width="1"/>
          <line x1="{pad_l}" y1="104" x2="{width - pad_r}" y2="104" stroke="#1e293b" stroke-width="1"/>

          {"".join(overlap_zones_svg)}
          {"".join(heading_svg)}
          {"".join(chunk_bars_svg)}

          <!-- Divider -->
          <line x1="10" y1="132" x2="{width - 10}" y2="132" stroke="#1e293b" stroke-width="1"/>
          <text x="10" y="158" fill="#94a3b8" font-size="10" font-family="monospace">Chunk Size</text>
          <text x="10" y="172" fill="#64748b" font-size="9.5" font-family="monospace">(chars)</text>

          <!-- Target max chunk size threshold line -->
          <line x1="{pad_l}" y1="{target_y:.1f}" x2="{width - pad_r}" y2="{target_y:.1f}" stroke="#f43f5e" stroke-width="1.2" stroke-dasharray="4 3"/>
          <text x="{width - pad_r}" y="{ max(144.0, target_y - 4):.1f}" fill="#fda4af" font-size="9.5" font-family="monospace" text-anchor="end">max_size={target_chunk_size}</text>
          <line x1="{pad_l}" y1="{hist_bottom}" x2="{width - pad_r}" y2="{hist_bottom}" stroke="#334155" stroke-width="1"/>

          {"".join(hist_bars)}
        </svg>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# 5. Notebook 3: Hybrid Search Stacked RRF Bars + RRF Decay Curve
# ---------------------------------------------------------------------------


def render_hybrid_rrf_svg(
    hybrid_results: list[HybridSearchResult],
    alpha: float,
    rrf_k: int,
) -> str:
    """Render a side-by-side interactive SVG showing:
    (Left) Stacked Horizontal Bar Chart decomposing each chunk's RRF_Score into
           Vector Contribution (teal) + BM25 Contribution (amber).
    (Right) The RRF Smoothing Curve 1 / (k + rank) for ranks 1..12."""
    if not hybrid_results:
        return ""

    # Compute max possible RRF score at rank 1 for scaling bars
    max_rrf = max(
        [1.0 / max(10, rrf_k)] + [r.rrf_score for r in hybrid_results]
    )

    bars_svg: list[str] = []
    row_h = 34
    left_w = 385
    bar_start_x = 135
    bar_max_w = 185

    for idx, r in enumerate(hybrid_results):
        y = 36 + idx * row_h
        vec_contrib = alpha / (rrf_k + r.vector_rank)
        bm25_contrib = max(0.0, r.rrf_score - vec_contrib)

        vec_w = (vec_contrib / max_rrf) * bar_max_w
        bm25_w = (bm25_contrib / max_rrf) * bar_max_w

        lbl = f"#{r.final_rank} Chunk {r.chunk_id}"
        bars_svg.append(
            f'<g>'
            f'<text x="10" y="{y + 13}" fill="#f8fafc" font-size="11" font-weight="bold" font-family="monospace">{_esc(lbl)}</text>'
            f'<text x="10" y="{y + 25}" fill="#94a3b8" font-size="9.5" font-family="monospace">V:#{r.vector_rank} B:#{r.bm25_rank}</text>'
            f'<rect x="{bar_start_x}" y="{y + 4}" width="{bar_max_w}" height="18" rx="4" fill="#1e293b"/>'
            f'<rect x="{bar_start_x}" y="{y + 4}" width="{vec_w:.1f}" height="18" rx="3" fill="#2dd4bf">'
            f'<title>Vector RRF part: {vec_contrib:.5f} (Vector Rank #{r.vector_rank}, cos={r.vector_score:.3f})</title></rect>'
            f'<rect x="{bar_start_x + vec_w:.1f}" y="{y + 4}" width="{bm25_w:.1f}" height="18" rx="3" fill="#f59e0b">'
            f'<title>BM25 RRF part: {bm25_contrib:.5f} (BM25 Rank #{r.bm25_rank}, bm25={r.bm25_score:.2f})</title></rect>'
            f'<text x="{bar_start_x + vec_w + bm25_w + 6:.1f}" y="{y + 17}" fill="#e2e8f0" font-size="10.5" font-family="monospace">{r.rrf_score:.4f}</text>'
            f'</g>'
        )

    # Right panel: RRF Decay Curve 1/(k + rank) for ranks 1..10
    curve_x0, curve_y0, curve_w, curve_h = 430, 42, 205, 175
    max_curve_y = 1.0 / (10.0 + 1.0)  # steepest when k=10, rank=1

    def curve_points(k_val: int) -> str:
        pts: list[str] = []
        for rnk in range(1, 11):
            px = curve_x0 + ((rnk - 1) / 9.0) * curve_w
            val = 1.0 / (k_val + rnk)
            py = (curve_y0 + curve_h) - (val / max_curve_y) * curve_h
            pts.append(f"{px:.1f},{py:.1f}")
        return " ".join(pts)

    ref_k10 = curve_points(10)
    ref_k60 = curve_points(60)
    active_pts = curve_points(rrf_k)

    return f"""
    <div style="background:#0f172a;padding:12px 14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;margin:10px 0;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
        <span style="font-size:13px;font-weight:700;color:#e2e8f0;">
          ⚖️ Interactive Reciprocal Rank Fusion (RRF) Decomposition (α = {alpha:.1f}, k = {rrf_k})
        </span>
        <span style="font-size:11px;">
          <span style="color:#2dd4bf;">█ Semantic Vector Contribution ({alpha:.0%})</span> ·
          <span style="color:#f59e0b;">█ Lexical BM25 Contribution ({1.0 - alpha:.0%})</span>
        </span>
      </div>
      <div style="overflow-x:auto;">
        <svg width="660" height="248" viewBox="0 0 660 248" style="background:#020617;border-radius:8px;border:1px solid #1e293b;width:100%;max-width:660px;">
          <text x="10" y="22" fill="#cbd5e1" font-size="11" font-weight="bold" font-family="monospace">Stacked RRF Score per Retrieved Chunk</text>
          {"".join(bars_svg)}

          <!-- Divider -->
          <line x1="395" y1="14" x2="395" y2="234" stroke="#1e293b" stroke-width="1"/>

          <!-- Right Panel: RRF Rank Discount Curve -->
          <text x="{curve_x0}" y="22" fill="#cbd5e1" font-size="11" font-weight="bold" font-family="monospace">RRF Rank Discount: 1 / (k + rank)</text>
          <line x1="{curve_x0}" y1="{curve_y0}" x2="{curve_x0}" y2="{curve_y0 + curve_h}" stroke="#334155" stroke-width="1"/>
          <line x1="{curve_x0}" y1="{curve_y0 + curve_h}" x2="{curve_x0 + curve_w}" y2="{curve_y0 + curve_h}" stroke="#334155" stroke-width="1"/>
          <text x="{curve_x0 + curve_w / 2:.1f}" y="{curve_y0 + curve_h + 20}" fill="#94a3b8" font-size="10" font-family="monospace" text-anchor="middle">Retriever Rank (1 → 10)</text>

          <!-- Reference curves k=10 and k=60 -->
          <polyline points="{ref_k10}" fill="none" stroke="#475569" stroke-width="1.2" stroke-dasharray="3 3"/>
          <text x="{curve_x0 + 32}" y="{curve_y0 + 18}" fill="#64748b" font-size="9.5" font-family="monospace">k=10 (steep)</text>
          <polyline points="{ref_k60}" fill="none" stroke="#475569" stroke-width="1.2" stroke-dasharray="3 3"/>
          <text x="{curve_x0 + curve_w - 5}" y="{curve_y0 + curve_h - 32}" fill="#64748b" font-size="9.5" font-family="monospace" text-anchor="end">k=60 (smooth)</text>

          <!-- Active curve -->
          <polyline points="{active_pts}" fill="none" stroke="#a855f7" stroke-width="3"/>
          <text x="{curve_x0 + curve_w - 5}" y="{curve_y0 + 30}" fill="#e9d5ff" font-size="10.5" font-weight="bold" font-family="monospace" text-anchor="end">Active k={rrf_k}</text>
        </svg>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# 6. Notebook 3: Stage-1 -> Stage-2 Reranking Slopegraph & Cross-Attention Matrix
# ---------------------------------------------------------------------------


def render_rerank_slopegraph_svg(
    all_reranked: list[RerankedCandidate],
    top_k: int,
) -> str:
    """Render a two-column Rank Slopegraph showing Stage-1 -> Stage-2 rank shifts."""
    if not all_reranked:
        return ""

    n = len(all_reranked)
    row_h = 34
    height = max(210, 56 + n * row_h)
    width = 660

    x_left = 205
    x_right = 445

    cutoff_y = 40 + top_k * row_h

    lines_svg: list[str] = []
    nodes_svg: list[str] = []

    for item in all_reranked:
        y1 = 42 + (item.stage1_rank - 1) * row_h + 12
        y2 = 42 + (item.stage2_rank - 1) * row_h + 12

        if item.rank_delta > 0:
            col = "#2dd4bf"  # Promoted
            delta_txt = f"+{item.rank_delta} ↑"
        elif item.rank_delta < 0:
            col = "#f43f5e"  # Demoted
            delta_txt = f"{item.rank_delta} ↓"
        else:
            col = "#94a3b8"  # Unchanged
            delta_txt = "0 ="

        in_top_k = item.stage2_rank <= top_k
        opacity = "0.95" if in_top_k or item.rank_delta != 0 else "0.45"
        stroke_w = "2.8" if in_top_k else "1.8"

        # Cubic bezier curve between Stage 1 and Stage 2
        cx1 = x_left + 85
        cx2 = x_right - 85
        lines_svg.append(
            f'<path d="M {x_left} {y1} C {cx1} {y1}, {cx2} {y2}, {x_right} {y2}" '
            f'fill="none" stroke="{col}" stroke-width="{stroke_w}" opacity="{opacity}">'
            f'<title>Chunk #{item.chunk_id} ({_esc(item.section_title)}): Stage-1 #{item.stage1_rank} → Stage-2 #{item.stage2_rank} (CE={item.rerank_score:.4f})</title></path>'
        )

        short_sec = item.section_title[:22] + (".." if len(item.section_title) > 22 else "")
        nodes_svg.append(
            f'<g>'
            f'<circle cx="{x_left}" cy="{y1}" r="5" fill="#64748b"/>'
            f'<text x="{x_left - 10}" y="{y1 + 4}" fill="#cbd5e1" font-size="10.5" font-family="monospace" text-anchor="end">'
            f'#{item.stage1_rank} Chunk {item.chunk_id} ({_esc(short_sec)})</text>'
            f'<circle cx="{x_right}" cy="{y2}" r="5.5" fill="{col}"/>'
            f'<text x="{x_right + 10}" y="{y2 + 4}" fill="{"#f8fafc" if in_top_k else "#64748b"}" '
            f'font-size="10.5" font-weight="{"bold" if in_top_k else "normal"}" font-family="monospace">'
            f'#{item.stage2_rank} Chunk {item.chunk_id} [CE={item.rerank_score:.3f} | {delta_txt}]</text>'
            f'</g>'
        )

    return f"""
    <div style="background:#0f172a;padding:12px 14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;margin:10px 0;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:8px;">
        <span style="font-size:13px;font-weight:700;color:#e2e8f0;">
          🔀 Stage-1 (Hybrid Recall) → Stage-2 (Cross-Encoder) Rank Slopegraph
        </span>
        <span style="font-size:11px;">
          <span style="color:#2dd4bf;">━ Promoted Direct Answers</span> ·
          <span style="color:#f43f5e;">━ Demoted Distractors</span> ·
          <span style="color:#38bdf8;">▭ Top-{top_k} Prompt Cutoff</span>
        </span>
      </div>
      <div style="overflow-x:auto;">
        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#020617;border-radius:8px;border:1px solid #1e293b;width:100%;max-width:{width}px;">
          <!-- Stage Headers -->
          <text x="{x_left - 10}" y="24" fill="#94a3b8" font-size="11" font-weight="bold" font-family="monospace" text-anchor="end">STAGE 1: Hybrid RRF Rank</text>
          <text x="{x_right + 10}" y="24" fill="#38bdf8" font-size="11" font-weight="bold" font-family="monospace">STAGE 2: Cross-Encoder Reranked</text>

          <!-- Top-K Selection Zone on Right -->
          <rect x="{x_right - 12}" y="36" width="{width - x_right + 4}" height="{top_k * row_h}" rx="6" fill="#0f766e" opacity="0.16" stroke="#14b8a6" stroke-dasharray="3 3"/>
          <line x1="20" y1="{cutoff_y}" x2="{width - 15}" y2="{cutoff_y}" stroke="#f43f5e" stroke-width="1.2" stroke-dasharray="5 4"/>
          <text x="{ (x_left + x_right) / 2:.1f}" y="{cutoff_y - 5}" fill="#fda4af" font-size="9.5" font-family="monospace" text-anchor="middle">✂️ Top-{top_k} Context Cutoff (Below this line is filtered out)</text>

          {"".join(lines_svg)}
          {"".join(nodes_svg)}
        </svg>
      </div>
    </div>
    """


def render_cross_attention_heatmap_svg(
    query: str,
    chunk_text: str,
    section_title: str,
    rerank_score: float,
    explanation: str,
) -> str:
    """Render a 2D Cross-Attention Token × Sentence interaction heatmap showing why
    a Cross-Encoder promotes passages with tight token clustering."""
    q_tokens = list(dict.fromkeys(tokenize(query, stem=False, drop_stopwords=True)))[:10]
    sentences = [
        s for s in split_sentences(chunk_text, min_len=9) if not s.startswith("#")
    ][:5]
    if not sentences:
        sentences = [chunk_text[:120]]
    if not q_tokens:
        q_tokens = ["query"]

    col_w = 54
    row_h = 28
    pad_l = 185
    pad_t = 52
    width = max(480, pad_l + len(q_tokens) * col_w + 30)
    height = pad_t + len(sentences) * row_h + 24

    headers_svg = [
        f'<text x="{pad_l + c_idx * col_w + col_w / 2:.1f}" y="38" fill="#38bdf8" font-size="10" font-family="monospace" text-anchor="middle">{_esc(qt[:7])}</text>'
        for c_idx, qt in enumerate(q_tokens)
    ]

    cells_svg: list[str] = []
    for r_idx, sent in enumerate(sentences):
        y = pad_t + r_idx * row_h
        s_set = set(tokenize(sent, stem=False, drop_stopwords=True))
        short_sent = sent[:24] + ("..." if len(sent) > 24 else "")
        cells_svg.append(
            f'<text x="10" y="{y + 17}" fill="#cbd5e1" font-size="10" font-family="monospace">'
            f'S{r_idx + 1}: {_esc(short_sent)}<title>{_esc(sent)}</title></text>'
        )
        for c_idx, qt in enumerate(q_tokens):
            x = pad_l + c_idx * col_w
            exact_hit = qt in s_set
            # Check substring / prefix overlap for partial interaction
            partial_hit = (not exact_hit) and any(
                st.startswith(qt[:4]) or qt.startswith(st[:4])
                for st in s_set
                if len(st) >= 4 and len(qt) >= 4
            )
            if exact_hit:
                fill = "#14b8a6"
                lbl = "1.0"
            elif partial_hit:
                fill = "#0369a1"
                lbl = "0.5"
            else:
                fill = "#0f172a"
                lbl = "·"

            cells_svg.append(
                f'<g>'
                f'<rect x="{x + 2}" y="{y + 2}" width="{col_w - 4}" height="{row_h - 4}" rx="4" fill="{fill}" stroke="#1e293b">'
                f'<title>Query token "{_esc(qt)}" × Sentence {r_idx + 1}: {"Exact Match" if exact_hit else ("Stem Match" if partial_hit else "No Match")}</title></rect>'
                f'<text x="{x + col_w / 2:.1f}" y="{y + 17}" fill="{"#020617" if exact_hit else "#64748b"}" font-size="10" font-weight="bold" font-family="monospace" text-anchor="middle">{lbl}</text>'
                f'</g>'
            )

    return f"""
    <div style="background:#0f172a;padding:12px 14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;margin-top:10px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
        <span style="font-size:13px;font-weight:700;color:#e2e8f0;">
          🎯 Cross-Encoder Token × Sentence Attention Matrix: <code>{_esc(section_title)}</code>
        </span>
        <span style="font-size:11px;font-family:monospace;color:#2dd4bf;">
          Score = {rerank_score:.4f} ({_esc(explanation)})
        </span>
      </div>
      <div style="overflow-x:auto;">
        <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#020617;border-radius:8px;border:1px solid #1e293b;">
          <text x="10" y="22" fill="#94a3b8" font-size="10" font-family="monospace">Passage Sentences ↓ \\ Query Tokens →</text>
          {"".join(headers_svg)}
          {"".join(cells_svg)}
        </svg>
      </div>
    </div>
    """


# ---------------------------------------------------------------------------
# 7. Notebook 3 & 4: Dynamic Mermaid Diagrams & Claim Verification Meters
# ---------------------------------------------------------------------------


def build_agentic_mermaid_diagram(trace: AgenticTrace) -> str:
    """Build a dynamic Mermaid flowchart string representing a live AgenticTrace."""
    def clean_mermaid_label(txt: str, max_len: int = 42) -> str:
        compact = re.sub(r'["\[\]{}()<>|#`]', "", txt).strip()
        if len(compact) > max_len:
            compact = compact[: max_len - 2] + ".."
        return compact

    lines = [
        "flowchart LR",
        f'    Q["User Query: {clean_mermaid_label(trace.original_query, 36)}"] --> Plan{{"Query Planner ({("Multi-Hop" if trace.is_multi_hop else "Single-Hop")})"}}',
    ]

    for step in trace.retrieval_steps:
        sq_id = f"SQ{step.step_number}"
        ch_id = f"CH{step.step_number}"
        sq_lbl = clean_mermaid_label(step.sub_query, 32)
        ch_lbl = f"Chunk #{step.top_chunk_id}: {clean_mermaid_label(step.top_section, 22)} ({step.rerank_score:.2f})"
        lines.append(f'    Plan --> {sq_id}["Step {step.step_number}: {sq_lbl}"]')
        lines.append(f'    {sq_id} --> {ch_id}[("{ch_lbl}")]')
        lines.append(f"    {ch_id} --> Gate")

    gate_status = "Pass" if trace.sufficiency_passed else "Retry Triggered"
    lines.append(f'    Gate{{"Sufficiency Gate: {gate_status}"}} --> LLM["Llama.app Synthesis ({len(trace.final_context_chunks)} Chunks)"]')
    lines.append(f'    LLM --> Audit["Faithfulness Audit: {trace.faithfulness_score:.0%}"]')
    return "\n".join(lines)


def render_faithfulness_audit_svg(trace: AgenticTrace) -> str:
    """Render visual confidence bars for each audited claim in an AgenticTrace."""
    verifications = trace.verifications
    if not verifications:
        return ""

    rows_html: list[str] = []
    for idx, v in enumerate(verifications):
        pct = max(4.0, min(100.0, v.confidence * 100.0))
        color = "#2dd4bf" if v.is_grounded else "#f43f5e"
        badge = (
            f'<span style="background:#064e3b;color:#6ee7b7;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:11px;">'
            f'✅ Cited Chunk #{v.supporting_chunk_id} ({v.confidence:.2f})</span>'
            if v.is_grounded
            else f'<span style="background:#4c0519;color:#fda4af;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:11px;">'
            f'⚠️ Unverified ({v.confidence:.2f})</span>'
        )
        rows_html.append(
            f"""
            <div style="margin-bottom:8px;background:#020617;padding:8px 12px;border-radius:8px;border:1px solid #1e293b;">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:5px;flex-wrap:wrap;">
                <span style="font-size:12px;color:#e2e8f0;flex:1;"><strong>Claim #{idx + 1}:</strong> {_esc(v.claim)}</span>
                {badge}
              </div>
              <div style="background:#1e293b;height:7px;border-radius:4px;position:relative;overflow:hidden;">
                <div style="position:absolute;left:35%;top:0;bottom:0;width:1.5px;background:#fbbf24;" title="Verification Threshold (0.35)"></div>
                <div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """
        )

    return f"""
    <div style="background:#0f172a;padding:12px 14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;margin-top:8px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-size:13px;font-weight:700;color:#e2e8f0;">
          🛡️ Claim-by-Claim Grounding & Faithfulness Meter (Overall: {trace.faithfulness_score:.0%})
        </span>
        <span style="font-size:11px;color:#fbbf24;font-family:monospace;">
          │ Amber Line = 0.35 Grounding Threshold
        </span>
      </div>
      {"".join(rows_html)}
    </div>
    """


def build_studio_pipeline_mermaid(
    chunk_strategy: str,
    alpha: float,
    use_reranker: bool,
    use_agentic: bool,
    top_k: int,
    total_chunks: int,
) -> str:
    """Build a dynamic Mermaid flowchart for Notebook 4 showing which production
    upgrades are currently active in Pipeline B."""
    chunk_lbl = "Markdown Section Chunker" if chunk_strategy == "markdown" else "Fixed-Char Sliding Window"
    hybrid_lbl = f"Hybrid RRF (α={alpha:.1f} Vec / {1.0 - alpha:.1f} BM25)"
    rerank_node = (
        f'Rerank["Stage-2 Cross-Encoder (Top-{top_k})"]'
        if use_reranker or use_agentic
        else f'Rerank["Reranker Bypassed (Direct Top-{top_k})"]'
    )
    agent_node = (
        'Plan{{"Agentic Multi-Hop Planner"}}'
        if use_agentic
        else 'Plan["Single-Shot Query"]'
    )
    verify_node = (
        'Verify["Faithfulness Audit & Citations"]'
        if use_agentic
        else 'Verify["Grounded Answer"]'
    )
    return "\n".join(
        [
            "flowchart LR",
            f'    Docs["Corpus + Custom Doc"] --> Chunker["{chunk_lbl}"]',
            f'    Chunker --> DB[("DuckDB Store ({total_chunks} Chunks)")]',
            f'    Q["User Question"] --> {agent_node}',
            f'    Plan --> Hybrid["{hybrid_lbl}"]',
            "    DB --> Hybrid",
            f"    Hybrid --> {rerank_node}",
            '    Rerank --> LLM["Llama.app Synthesis"]',
            f"    LLM --> {verify_node}",
        ]
    )


def render_ab_arena_comparison_svg(
    baseline_hits: list[tuple[str, float]],
    prod_chunks: list[tuple[str, float, str]],
) -> str:
    """Render a visual side-by-side relevance bar comparison between Pipeline A
    (Scratch Vector RAG) and Pipeline B (Production RAG) for Notebook 4."""
    rows_a: list[str] = []
    for i, (txt, sc) in enumerate(baseline_hits):
        pct = max(5.0, min(100.0, sc * 100.0))
        snippet = txt[:68].replace("\n", " ") + ("..." if len(txt) > 68 else "")
        rows_a.append(
            f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:3px;">
                <span style="color:#cbd5e1;font-family:monospace;"><strong>#{i + 1}</strong> {_esc(snippet)}</span>
                <span style="color:#38bdf8;font-family:monospace;font-weight:700;">cos={sc:.3f}</span>
              </div>
              <div style="background:#1e293b;height:8px;border-radius:4px;overflow:hidden;">
                <div style="width:{pct:.1f}%;background:#38bdf8;height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """
        )

    rows_b: list[str] = []
    for i, (lbl, sc, txt) in enumerate(prod_chunks):
        # Scale RRF scores (typically ~0.016) or Cross-Encoder scores (0..1) appropriately
        scaled_pct = max(8.0, min(100.0, (sc * 100.0) if sc > 0.05 else (sc * 3200.0)))
        rows_b.append(
            f"""
            <div style="margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:3px;">
                <span style="color:#e2e8f0;font-family:monospace;"><strong>#{i + 1}</strong> {_esc(lbl[:54])}</span>
                <span style="color:#2dd4bf;font-family:monospace;font-weight:700;">score={sc:.3f}</span>
              </div>
              <div style="background:#1e293b;height:8px;border-radius:4px;overflow:hidden;">
                <div style="width:{scaled_pct:.1f}%;background:linear-gradient(90deg,#14b8a6,#2dd4bf);height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """
        )

    return f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;background:#0f172a;padding:12px 14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;margin:10px 0;">
      <div style="background:#020617;padding:10px 12px;border-radius:8px;border:1px solid #1e293b;">
        <div style="font-size:12.5px;font-weight:700;color:#38bdf8;margin-bottom:8px;">
          📊 Pipeline A: Scratch Vector Retrieval Scores
        </div>
        {"".join(rows_a)}
      </div>
      <div style="background:#020617;padding:10px 12px;border-radius:8px;border:1px solid #1e293b;">
        <div style="font-size:12.5px;font-weight:700;color:#2dd4bf;margin-bottom:8px;">
          🚀 Pipeline B: Production Retrieval Scores
        </div>
        {"".join(rows_b)}
      </div>
    </div>
    """
