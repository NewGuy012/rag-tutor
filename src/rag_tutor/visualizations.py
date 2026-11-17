"""Interactive SVG, Heatmap, Embedding Projection, and Mermaid Visualization Builders."""

from __future__ import annotations

import html
import math
import re
from typing import Sequence

from .agentic_rag import AgenticTrace
from .chunker import Chunk, extract_markdown_headings
from .duckdb_store import DuckDBChunkRecord
from .hybrid_search import HybridSearchResult
from .llm_backend import SEMANTIC_CLUSTERS, split_sentences, tokenize
from .modular_graph_rag import (
    GraphSearchResult,
    ModularRAGConfig,
    QueryExpansionResult,
)
from .reranker import RerankedCandidate
from .scratch_rag import cosine_similarity, dot_product, vector_norm


def _esc(text: str) -> str:
    """HTML-escape text for safe inclusion in SVG/HTML."""
    return html.escape(str(text), quote=True)


def _trunc(text: str, max_len: int, ellipsis: str = "..") -> str:
    """Truncate `text` to `max_len` characters with `ellipsis`."""
    return text[: max_len - len(ellipsis)] + ellipsis if len(text) > max_len else text


def _clean_mermaid_label(txt: str, max_len: int = 42) -> str:
    """Sanitize a label string for Mermaid node text."""
    compact = re.sub(r'["\[\]{}()<>|#`\\;\r\n\t]+', " ", str(txt))
    return _trunc(re.sub(r"\s+", " ", compact).strip(), max_len)


def _card_shell(
    title: str,
    subtitle: str,
    body: str,
    footer: str = "",
    margin: str = "10px 0",
) -> str:
    """Wrap a visualization inside a dark-slate responsive card."""
    return (
        f'<div style="background:#0f172a;padding:12px 14px;border-radius:12px;'
        f'border:1px solid #334155;color:#f8fafc;margin:{margin};">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:8px;flex-wrap:wrap;gap:8px;">'
        f'<span style="font-size:13px;font-weight:700;color:#e2e8f0;">{title}</span>'
        f'<span style="font-size:11px;color:#cbd5e1;">{subtitle}</span>'
        f"</div>"
        f'<div style="overflow-x:auto;">{body}</div>'
        f"{footer}"
        f"</div>"
    )


def _svg_frame(width: int, height: int, inner: str, responsive: bool = True) -> str:
    """Wrap SVG elements in a styled dark canvas."""
    extra = f"width:100%;max-width:{width}px;" if responsive else ""
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="background:#020617;border-radius:8px;border:1px solid #1e293b;{extra}">'
        f"{inner}</svg>"
    )


def _metric_row(
    left_html: str, right_html: str, pct: float, fill: str, *,
    bar_h: int = 8, marker_pct: float | None = None, marker_color: str = "#fbbf24",
    marker_title: str = "", boxed: bool = False, sub_html: str = "",
) -> str:
    title_attr = f' title="{_esc(marker_title)}"' if marker_title else ""
    marker = (
        f'<div style="position:absolute;left:{marker_pct:.1f}%;top:0;bottom:0;width:1.5px;background:{marker_color};"{title_attr}></div>'
        if marker_pct is not None else ""
    )
    bar = (
        f'<div style="background:#1e293b;height:{bar_h}px;border-radius:4px;position:relative;overflow:hidden;">'
        f'{marker}<div style="width:{pct:.1f}%;background:{fill};height:100%;border-radius:4px;"></div></div>'
    )
    inner = (
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:8px;font-size:12px;margin-bottom:4px;flex-wrap:wrap;">'
        f'{left_html}{right_html}</div>{bar}{sub_html}'
    )
    if boxed:
        return f'<div style="background:#020617;padding:8px 12px;border-radius:8px;border:1px solid #1e293b;margin-bottom:8px;">{inner}</div>'
    return f'<div style="margin-bottom:10px;">{inner}</div>'


def _stacked_bar_row(
    y: int,
    x_lbl: int,
    x_bar: int,
    bar_w: int,
    x_score: int,
    title: str,
    subtitle: str,
    sub_color: str,
    vw: float,
    bw: float,
    score_txt: str,
    vec_tip: str = "",
    bm25_tip: str = "",
) -> str:
    """Render one stacked Dense (teal) + BM25 (amber) horizontal bar row."""
    v_tip = f"<title>{_esc(vec_tip)}</title>" if vec_tip else ""
    b_tip = f"<title>{_esc(bm25_tip)}</title>" if bm25_tip else ""
    return (
        f"<g>"
        f'<text x="{x_lbl}" y="{y + 13}" fill="#f8fafc" font-size="10.5" font-weight="bold" font-family="monospace">{_esc(title)}</text>'
        f'<text x="{x_lbl}" y="{y + 24}" fill="{sub_color}" font-size="9" font-family="monospace">{_esc(subtitle)}</text>'
        f'<rect x="{x_bar}" y="{y + 4}" width="{bar_w}" height="17" rx="4" fill="#1e293b"/>'
        f'<rect x="{x_bar}" y="{y + 4}" width="{vw:.1f}" height="17" rx="3" fill="#2dd4bf">{v_tip}</rect>'
        f'<rect x="{x_bar + vw:.1f}" y="{y + 4}" width="{bw:.1f}" height="17" rx="3" fill="#f59e0b">{b_tip}</rect>'
        f'<text x="{x_score}" y="{y + 16}" fill="#e2e8f0" font-size="10" font-family="monospace">{_esc(score_txt)}</text>'
        f"</g>"
    )


def _rrf_bar_parts(
    result: HybridSearchResult,
    alpha: float,
    rrf_k: int,
    max_rrf: float,
    bar_w: float,
) -> tuple[float, float, float, float]:
    """Compute `(vec_contrib, bm25_contrib, vec_width, bm25_width)` for stacked RRF bars."""
    vec_contrib = alpha / (rrf_k + result.vector_rank)
    bm25_contrib = max(0.0, result.rrf_score - vec_contrib)
    return (
        vec_contrib,
        bm25_contrib,
        (vec_contrib / max_rrf) * bar_w,
        (bm25_contrib / max_rrf) * bar_w,
    )


def render_2d_compass_svg(
    vec_a: Sequence[float],
    vec_b: Sequence[float],
    dot_val: float,
    norm_a: float,
    norm_b: float,
    cos_val: float,
    angle_deg: float,
) -> str:
    """Render an interactive 2D vector compass with metric comparison gauges."""
    cx, cy, scale = 150.0, 150.0, 108.0
    ax, ay = cx + vec_a[0] * scale, cy - vec_a[1] * scale
    bx, by = cx + vec_b[0] * scale, cy - vec_b[1] * scale

    proj_line_svg = ""
    arc_svg = ""
    if norm_a > 1e-6 and norm_b > 1e-6:
        ua_x, ua_y = vec_a[0] / norm_a, vec_a[1] / norm_a
        proj_len = dot_val / norm_a
        px, py = cx + (ua_x * proj_len) * scale, cy - (ua_y * proj_len) * scale
        proj_line_svg = (
            f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
            f'stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3 3" />'
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" fill="#94a3b8" />'
        )
        arc_r = 38.0
        ang_a, ang_b = math.atan2(-vec_a[1], vec_a[0]), math.atan2(-vec_b[1], vec_b[0])
        sx, sy = cx + arc_r * math.cos(ang_a), cy + arc_r * math.sin(ang_a)
        ex, ey = cx + arc_r * math.cos(ang_b), cy + arc_r * math.sin(ang_b)
        diff = (ang_b - ang_a + math.pi) % (2 * math.pi) - math.pi
        sweep = 1 if diff >= 0 else 0
        mid_ang = ang_a + diff / 2.0
        lx, ly = cx + (arc_r + 16) * math.cos(mid_ang), cy + (arc_r + 16) * math.sin(mid_ang)
        arc_svg = (
            f'<path d="M {sx:.1f} {sy:.1f} A {arc_r:.1f} {arc_r:.1f} 0 0 {sweep} {ex:.1f} {ey:.1f}" '
            f'fill="none" stroke="#a855f7" stroke-width="2.2" />'
            f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#e9d5ff" font-size="11" '
            f'font-family="monospace" text-anchor="middle" dominant-baseline="middle">{angle_deg:.0f}°</text>'
        )

    euclid_dist = math.sqrt((vec_a[0] - vec_b[0]) ** 2 + (vec_a[1] - vec_b[1]) ** 2)
    euclid_sim = 1.0 / (1.0 + euclid_dist)
    cos_pct = max(0.0, min(100.0, (cos_val + 1.0) * 50.0))
    dot_pct = max(0.0, min(100.0, (max(-2.0, min(2.0, dot_val)) + 2.0) * 25.0))
    euc_pct = max(0.0, min(100.0, euclid_sim * 100.0))

    if cos_val >= 0.75:
        verdict, verdict_color = "🟢 High Semantic Alignment (Same Direction)", "#2dd4bf"
    elif cos_val >= 0.25:
        verdict, verdict_color = "🟡 Moderate Similarity (Acute Angle)", "#fbbf24"
    elif cos_val >= -0.25:
        verdict, verdict_color = "⚪ Near-Orthogonal (~90° Unrelated)", "#94a3b8"
    else:
        verdict, verdict_color = "🔴 Opposite Semantic Direction (< 0)", "#f87171"

    def _gauge(title_html: str, val_html: str, val_color: str, pct: float, bar_bg: str, mid_tick: bool = False) -> str:
        return _metric_row(
            f'<span style="color:#cbd5e1;">{title_html}</span>',
            f'<span style="font-family:monospace;color:{val_color};font-weight:700;">{val_html}</span>',
            pct, bar_bg, bar_h=10, marker_pct=50.0 if mid_tick else None, marker_color="#475569",
        )

    gauges = "".join([
        _gauge("<strong>Cosine Similarity</strong> (Angle-only, scale-invariant)", f"{cos_val:+.4f} (θ = {angle_deg:.1f}°)", "#2dd4bf", cos_pct, "linear-gradient(90deg,#f43f5e,#eab308,#2dd4bf)", True),
        _gauge("<strong>Raw Dot Product (A · B)</strong> (Sensitive to vector length)", f"{dot_val:+.4f}", "#f59e0b", dot_pct, "#f59e0b"),
        _gauge("<strong>Euclidean Distance ‖A − B‖</strong> (Tip-to-tip distance)", f"dist = {euclid_dist:.3f} (sim = {euclid_sim:.3f})", "#c084fc", euc_pct, "#c084fc"),
    ])

    return f"""
    <div style="display:flex;flex-wrap:wrap;gap:18px;align-items:center;background:#0f172a;padding:14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;">
      <svg width="300" height="300" viewBox="0 0 300 300" style="background:#020617;border-radius:10px;border:1px solid #1e293b;flex-shrink:0;">
        <circle cx="{cx}" cy="{cy}" r="{scale * 0.5:.1f}" fill="none" stroke="#1e293b" stroke-dasharray="3 3"/>
        <circle cx="{cx}" cy="{cy}" r="{scale:.1f}" fill="none" stroke="#334155" stroke-dasharray="4 4"/>
        <line x1="16" y1="{cy}" x2="284" y2="{cy}" stroke="#334155" stroke-width="1"/>
        <line x1="{cx}" y1="16" x2="{cx}" y2="284" stroke="#334155" stroke-width="1"/>
        <text x="278" y="{cy - 6}" fill="#64748b" font-size="10" font-family="monospace" text-anchor="end">+x</text>
        <text x="{cx + 6}" y="26" fill="#64748b" font-size="10" font-family="monospace">+y</text>
        {proj_line_svg}
        {arc_svg}
        <line x1="{cx}" y1="{cy}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="#2dd4bf" stroke-width="4" stroke-linecap="round"/>
        <circle cx="{ax:.1f}" cy="{ay:.1f}" r="6" fill="#2dd4bf"/>
        <text x="{ax + 8:.1f}" y="{ay - 8:.1f}" fill="#2dd4bf" font-size="12" font-weight="bold" font-family="monospace">A ({vec_a[0]:.2f}, {vec_a[1]:.2f})</text>
        <line x1="{cx}" y1="{cy}" x2="{bx:.1f}" y2="{by:.1f}" stroke="#f59e0b" stroke-width="4" stroke-linecap="round"/>
        <circle cx="{bx:.1f}" cy="{by:.1f}" r="6" fill="#f59e0b"/>
        <text x="{bx + 8:.1f}" y="{by + 15:.1f}" fill="#f59e0b" font-size="12" font-weight="bold" font-family="monospace">B ({vec_b[0]:.2f}, {vec_b[1]:.2f})</text>
      </svg>
      <div style="flex:1;min-width:260px;font-family:system-ui,-apple-system,sans-serif;">
        <div style="font-size:14px;font-weight:700;color:{verdict_color};margin-bottom:10px;">{verdict}</div>
        {gauges}
        <div style="font-size:12px;color:#94a3b8;background:#020617;padding:8px 10px;border-radius:6px;border-left:3px solid #a855f7;">
          💡 <strong>Visual Intuition:</strong> The dashed gray line shows the orthogonal projection of <strong>B</strong> onto <strong>A</strong>.
          Try scaling <strong>A</strong> from <code>(0.9, 0.0)</code> down to <code>(0.2, 0.0)</code>:
          <em>Dot Product</em> and <em>Euclidean Distance</em> change, but <strong>Cosine Similarity stays constant!</strong>
        </div>
      </div>
    </div>
    """


def render_embedding_barcode_svg(
    query_vec: Sequence[float],
    doc_vec: Sequence[float],
    cosine_score: float,
) -> str:
    """Render a 3-row 128-D embedding barcode comparing Query, Chunk, and Q_i * D_i."""
    dim = min(len(query_vec), len(doc_vec))
    if dim == 0:
        return ""

    cell_w, x_offset = 5.2, 125
    width = int(dim * cell_w + 135)
    num_clusters = min(len(SEMANTIC_CLUSTERS), dim // 3)
    lex_end = max(num_clusters + 16, int(dim * 0.6))

    def val_to_color(val: float, is_product: bool = False) -> str:
        t = min(1.0, abs(val) * (18.0 if is_product else 4.2))
        if is_product:
            tr, tg, tb = (45, 212, 191) if val >= 0 else (244, 63, 94)
        else:
            tr, tg, tb = (56, 189, 248) if val >= 0 else (251, 146, 60)
        return f"rgb({int(15 + (tr - 15) * t)},{int(23 + (tg - 23) * t)},{int(42 + (tb - 42) * t)})"

    rects: list[str] = []
    contributions: list[tuple[float, str]] = []
    for i in range(dim):
        qv, dv = query_vec[i], doc_vec[i]
        prod = qv * dv
        x = x_offset + i * cell_w
        if i < num_clusters:
            dim_name = f"Dim {i} (Semantic: {SEMANTIC_CLUSTERS[i][0]})"
        elif i < lex_end:
            dim_name = f"Dim {i} (Lexical Token Hash)"
        else:
            dim_name = f"Dim {i} (Char 3-Gram Subword)"
        contributions.append((prod, dim_name))
        for y, val, tag, is_prod in ((28, qv, "Q", False), (56, dv, "D", False), (84, prod, "Q*D", True)):
            rects.append(
                f'<rect x="{x:.1f}" y="{y}" width="{cell_w - 0.6:.1f}" height="20" '
                f'fill="{val_to_color(val, is_prod)}"><title>{_esc(dim_name)}: {tag}[{i}] = {val:+.4f}</title></rect>'
            )

    contributions.sort(key=lambda item: item[0], reverse=True)
    top_badges = " ".join(
        f'<span style="background:#134e4a;color:#5eead4;padding:2px 8px;border-radius:999px;font-size:11px;font-family:monospace;">'
        f"{_esc(lbl)} (+{sc:.3f})</span>"
        for sc, lbl in contributions[:4]
        if sc > 0.001
    ) or '<span style="color:#94a3b8;font-size:11px;">Low dimensional overlap</span>'

    sem_end_x = x_offset + num_clusters * cell_w
    lex_end_x = x_offset + lex_end * cell_w
    total_end_x = x_offset + dim * cell_w
    svg = _svg_frame(
        width,
        122,
        f"""
        <text x="{(x_offset + sem_end_x) / 2:.1f}" y="16" fill="#38bdf8" font-size="10" font-family="monospace" text-anchor="middle">0..{num_clusters - 1} Semantic</text>
        <line x1="{sem_end_x:.1f}" y1="6" x2="{sem_end_x:.1f}" y2="112" stroke="#334155" stroke-dasharray="2 2"/>
        <text x="{(sem_end_x + lex_end_x) / 2:.1f}" y="16" fill="#a78bfa" font-size="10" font-family="monospace" text-anchor="middle">{num_clusters}..{lex_end - 1} Lexical Hash Projections</text>
        <line x1="{lex_end_x:.1f}" y1="6" x2="{lex_end_x:.1f}" y2="112" stroke="#334155" stroke-dasharray="2 2"/>
        <text x="{(lex_end_x + total_end_x) / 2:.1f}" y="16" fill="#f59e0b" font-size="10" font-family="monospace" text-anchor="middle">{lex_end}..{dim - 1} Char 3-Grams</text>
        <text x="10" y="42" fill="#38bdf8" font-size="11" font-family="monospace" font-weight="bold">Query ({dim}-D)</text>
        <text x="10" y="70" fill="#cbd5e1" font-size="11" font-family="monospace" font-weight="bold">Chunk ({dim}-D)</text>
        <text x="10" y="98" fill="#2dd4bf" font-size="11" font-family="monospace" font-weight="bold">Product Qᵢ×Dᵢ</text>
        {"".join(rects)}
        """,
        responsive=False,
    )
    footer = (
        f'<div style="margin-top:8px;font-size:12px;color:#cbd5e1;display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
        f"<span><strong>Top Aligned Dimensions:</strong></span> {top_badges}</div>"
    )
    return _card_shell(
        "🔬 128-D Embedding Barcode Inspector (Hover any bar to inspect dimension)",
        f'<span style="font-family:monospace;font-size:12px;color:#2dd4bf;">∑ (Qᵢ × Dᵢ) = Cosine Similarity = <strong>{cosine_score:.4f}</strong></span>',
        svg,
        footer=footer,
        margin="8px 0 0 0",
    )


TOPIC_PALETTE: list[tuple[str, str, set[str]]] = [
    ("Battery & Chemistry", "#c084fc", {"battery", "lfp", "nmc", "solid-state", "electrolyte", "density"}),
    ("Charging & 800V/V2H", "#38bdf8", {"charging", "800-volt", "nacs", "preconditioning", "bidirectional", "v2h", "soc"}),
    ("Motors & Regen Braking", "#2dd4bf", {"regenerative", "braking", "motor", "torque", "pmsm", "induction", "pedal"}),
    ("Efficiency & Winter Thermal", "#fbbf24", {"mpge", "efficiency", "aerodynamic", "heat", "pump", "winter", "cold"}),
    ("History, Safety & BMS", "#fb7185", {"1899", "jamais", "apollo", "lunar", "bms", "pyrotechnic", "12-volt"}),
]


def _classify_ev_topic(text: str) -> tuple[str, str]:
    tokens = set(tokenize(text))
    best = max(TOPIC_PALETTE, key=lambda item: len(tokens & item[2]))
    return best[0], best[1]


def _project_2d_deterministic(
    vectors: list[list[float]],
    query_vec: list[float],
) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Project vectors into 2D via centered orthogonal power iteration."""
    n = len(vectors)
    if n == 0:
        return [], (0.0, 0.0)
    dim = len(vectors[0])

    mean_vec = [sum(v[d] for v in vectors) / n for d in range(dim)]
    centered = [[v[d] - mean_vec[d] for d in range(dim)] for v in vectors]
    q_centered = [query_vec[d] - mean_vec[d] for d in range(dim)]

    def power_axis(deflate_axis: list[float] | None = None) -> list[float]:
        w = [math.sin((d + 1) * 1.7) if deflate_axis is None else math.cos((d + 1) * 2.3) for d in range(dim)]
        for _ in range(18):
            new_w = [0.0] * dim
            for row in centered:
                dot = dot_product(row, w)
                for d in range(dim):
                    new_w[d] += dot * row[d]
            if deflate_axis is not None:
                proj = dot_product(new_w, deflate_axis)
                new_w = [new_w[d] - proj * deflate_axis[d] for d in range(dim)]
            norm = vector_norm(new_w) or 1.0
            w = [x / norm for x in new_w]
        return w

    pc1 = power_axis(None)
    pc2 = power_axis(pc1)
    pts = [(dot_product(row, pc1), dot_product(row, pc2)) for row in centered]
    q_pt = (dot_product(q_centered, pc1), dot_product(q_centered, pc2))
    return pts, q_pt


def render_embedding_space_2d_svg(
    chunks: list[str],
    embeddings: list[list[float]],
    query_text: str,
    query_vec: list[float],
    top_n: int = 3,
) -> str:
    """Render an interactive 2D PCA projection of corpus embeddings and query rays."""
    pts, q_pt = _project_2d_deterministic(embeddings, query_vec)
    if not pts:
        return ""

    scored = sorted(
        ((idx, cosine_similarity(query_vec, emb)) for idx, emb in enumerate(embeddings)),
        key=lambda item: item[1],
        reverse=True,
    )
    top_map = {idx: (rank + 1, sim) for rank, (idx, sim) in enumerate(scored[:top_n])}
    all_sim_map = dict(scored)

    all_x = [p[0] for p in pts] + [q_pt[0]]
    all_y = [p[1] for p in pts] + [q_pt[1]]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x, span_y = max(0.05, max_x - min_x), max(0.05, max_y - min_y)

    width, height, pad = 640, 310, 42

    def to_screen(pt: tuple[float, float]) -> tuple[float, float]:
        return (
            pad + ((pt[0] - min_x) / span_x) * (width - 2 * pad),
            height - pad - ((pt[1] - min_y) / span_y) * (height - 2 * pad),
        )

    qx, qy = to_screen(q_pt)
    rays_svg: list[str] = []
    nodes_svg: list[str] = []

    for idx, (pt, chunk_text) in enumerate(zip(pts, chunks)):
        cx, cy = to_screen(pt)
        topic_lbl, color = _classify_ev_topic(chunk_text)
        sim_val = all_sim_map[idx]
        if idx in top_map:
            rank, _ = top_map[idx]
            rays_svg.append(
                f'<line x1="{qx:.1f}" y1="{qy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" '
                f'stroke="#2dd4bf" stroke-width="{max(1.5, 3.2 - rank * 0.4):.1f}" stroke-dasharray="4 3" opacity="0.85"/>'
            )
            nodes_svg.append(
                f'<g><circle cx="{cx:.1f}" cy="{cy:.1f}" r="11" fill="none" stroke="#2dd4bf" stroke-width="2" />'
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{color}" stroke="#020617" stroke-width="1.5">'
                f'<title>Top #{rank} (cos={sim_val:.4f}) · {_esc(topic_lbl)}&#10;{_esc(chunk_text)}</title></circle>'
                f'<text x="{cx + 12:.1f}" y="{cy + 4:.1f}" fill="#f8fafc" font-size="11" font-weight="bold" font-family="monospace">'
                f"#{rank} ({sim_val:.2f})</text></g>"
            )
        else:
            nodes_svg.append(
                f'<g><circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="{color}" opacity="0.72" stroke="#0f172a" stroke-width="1">'
                f'<title>Fact #{idx + 1} (cos={sim_val:.4f}) · {_esc(topic_lbl)}&#10;{_esc(chunk_text)}</title></circle></g>'
            )

    legend_items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;color:#cbd5e1;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{col};display:inline-block;"></span>'
        f"{_esc(lbl)}</span>"
        for lbl, col, _ in TOPIC_PALETTE
    )
    svg = _svg_frame(
        width,
        height,
        f"""
        <line x1="{width / 2:.1f}" y1="15" x2="{width / 2:.1f}" y2="{height - 15}" stroke="#1e293b" stroke-dasharray="3 3"/>
        <line x1="15" y1="{height / 2:.1f}" x2="{width - 15}" y2="{height / 2:.1f}" stroke="#1e293b" stroke-dasharray="3 3"/>
        <text x="{width - 12}" y="{height - 8}" fill="#475569" font-size="10" font-family="monospace" text-anchor="end">PC1 (Semantic Axis 1)</text>
        <text x="12" y="18" fill="#475569" font-size="10" font-family="monospace">PC2 (Semantic Axis 2)</text>
        {"".join(rays_svg)}
        {"".join(nodes_svg)}
        <circle cx="{qx:.1f}" cy="{qy:.1f}" r="14" fill="#2dd4bf" opacity="0.22"/>
        <circle cx="{qx:.1f}" cy="{qy:.1f}" r="7.5" fill="#14b8a6" stroke="#ffffff" stroke-width="2">
          <title>Live User Query: {_esc(query_text)}</title>
        </circle>
        <text x="{qx:.1f}" y="{qy - 14:.1f}" fill="#5eead4" font-size="12" font-weight="bold" font-family="monospace" text-anchor="middle">★ YOUR QUERY</text>
        """,
    )
    footer = f'<div style="display:flex;flex-wrap:wrap;gap:14px;margin-top:8px;padding-top:6px;border-top:1px solid #1e293b;">{legend_items}</div>'
    return _card_shell(
        "🌌 2D Embedding Space Map (PCA Projection of 128-D Vectors + Live Query Neighborhood)",
        f'<span style="color:#94a3b8;">Hover any point to read the fact · Dashed teal beams connect your Query (★) to Top-{top_n} matches</span>',
        svg,
        footer=footer,
        margin="0",
    )


def render_chunk_timeline_svg(
    raw_text: str,
    chunks: list[Chunk],
    strategy_name: str,
    target_chunk_size: int,
) -> str:
    """Render a document chunking minimap, overlap zones, and length histogram."""
    total_chars = max(1, len(raw_text))
    width, pad_l, pad_r = 660, 75, 25
    track_w = width - pad_l - pad_r
    palette = ["#38bdf8", "#60a5fa", "#2dd4bf", "#c084fc", "#f472b6"]
    lane_y = [48, 72, 96]

    heading_svg = [
        f'<line x1="{hx:.1f}" y1="22" x2="{hx:.1f}" y2="122" stroke="#64748b" stroke-width="1" stroke-dasharray="2 2">'
        f'<title>Heading @ char {h_pos}: {_esc(h_title)}</title></line>'
        f'<circle cx="{hx:.1f}" cy="22" r="3" fill="#94a3b8"><title>Heading: {_esc(h_title)}</title></circle>'
        for h_pos, h_title in extract_markdown_headings(raw_text)
        for hx in (pad_l + (h_pos / total_chars) * track_w,)
    ]

    chunk_bars_svg: list[str] = []
    overlap_zones_svg: list[str] = []
    for idx, ch in enumerate(chunks):
        x1 = pad_l + (max(0, ch.start_char) / total_chars) * track_w
        x2 = pad_l + (min(total_chars, ch.end_char) / total_chars) * track_w
        bw = max(6.0, x2 - x1)
        y = lane_y[idx % len(lane_y)]
        col = palette[idx % len(palette)]
        chunk_bars_svg.append(
            f'<g><rect x="{x1:.1f}" y="{y}" width="{bw:.1f}" height="16" rx="4" fill="{col}" opacity="0.88" stroke="#020617" stroke-width="1">'
            f'<title>Chunk #{ch.chunk_id} ({_esc(ch.section_title)}) · chars {ch.start_char}..{ch.end_char} ({ch.char_length} chars, overlap={ch.overlap_chars})</title></rect>'
            f'<text x="{x1 + bw / 2:.1f}" y="{y + 11.5:.1f}" fill="#020617" font-size="10" font-weight="bold" font-family="monospace" text-anchor="middle">#{ch.chunk_id}</text></g>'
        )
        if idx > 0 and ch.start_char < chunks[idx - 1].end_char:
            prev = chunks[idx - 1]
            ox1 = pad_l + (ch.start_char / total_chars) * track_w
            ox2 = pad_l + (min(total_chars, prev.end_char) / total_chars) * track_w
            overlap_zones_svg.append(
                f'<rect x="{ox1:.1f}" y="38" width="{max(3.0, ox2 - ox1):.1f}" height="80" fill="#f59e0b" opacity="0.26">'
                f'<title>Sliding Overlap between Chunk #{prev.chunk_id} and #{ch.chunk_id} ({ch.overlap_chars} chars)</title></rect>'
            )

    max_len = max([target_chunk_size] + [c.char_length for c in chunks] + [100])
    hist_top, hist_h = 148, 68
    hist_bottom = hist_top + hist_h
    target_y = hist_bottom - (target_chunk_size / max_len) * hist_h
    slot_w = track_w / max(1, len(chunks))
    bar_w = max(8.0, min(38.0, slot_w * 0.72))

    hist_bars = [
        f'<g><rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3" fill="{palette[idx % len(palette)]}">'
        f'<title>Chunk #{ch.chunk_id}: {ch.char_length} chars ({ch.word_count} words)</title></rect>'
        f'<text x="{bx + bar_w / 2:.1f}" y="{by - 4:.1f}" fill="#cbd5e1" font-size="9.5" font-family="monospace" text-anchor="middle">{ch.char_length}</text>'
        f'<text x="{bx + bar_w / 2:.1f}" y="{hist_bottom + 12:.1f}" fill="#94a3b8" font-size="9.5" font-family="monospace" text-anchor="middle">#{ch.chunk_id}</text></g>'
        for idx, ch in enumerate(chunks)
        for bx in (pad_l + idx * slot_w + (slot_w - bar_w) / 2.0,)
        for bh in (max(4.0, (ch.char_length / max_len) * hist_h),)
        for by in (hist_bottom - bh,)
    ]

    svg = _svg_frame(
        width,
        236,
        f"""
        <text x="10" y="25" fill="#94a3b8" font-size="10" font-family="monospace">Headings</text>
        <text x="10" y="60" fill="#cbd5e1" font-size="10" font-family="monospace">Swimlane 1</text>
        <text x="10" y="84" fill="#cbd5e1" font-size="10" font-family="monospace">Swimlane 2</text>
        <text x="10" y="108" fill="#cbd5e1" font-size="10" font-family="monospace">Swimlane 3</text>
        <line x1="{pad_l}" y1="56" x2="{width - pad_r}" y2="56" stroke="#1e293b" stroke-width="1"/>
        <line x1="{pad_l}" y1="80" x2="{width - pad_r}" y2="80" stroke="#1e293b" stroke-width="1"/>
        <line x1="{pad_l}" y1="104" x2="{width - pad_r}" y2="104" stroke="#1e293b" stroke-width="1"/>
        {"".join(overlap_zones_svg)}
        {"".join(heading_svg)}
        {"".join(chunk_bars_svg)}
        <line x1="10" y1="132" x2="{width - 10}" y2="132" stroke="#1e293b" stroke-width="1"/>
        <text x="10" y="158" fill="#94a3b8" font-size="10" font-family="monospace">Chunk Size</text>
        <text x="10" y="172" fill="#64748b" font-size="9.5" font-family="monospace">(chars)</text>
        <line x1="{pad_l}" y1="{target_y:.1f}" x2="{width - pad_r}" y2="{target_y:.1f}" stroke="#f43f5e" stroke-width="1.2" stroke-dasharray="4 3"/>
        <text x="{width - pad_r}" y="{max(144.0, target_y - 4):.1f}" fill="#fda4af" font-size="9.5" font-family="monospace" text-anchor="end">max_size={target_chunk_size}</text>
        <line x1="{pad_l}" y1="{hist_bottom}" x2="{width - pad_r}" y2="{hist_bottom}" stroke="#334155" stroke-width="1"/>
        {"".join(hist_bars)}
        """,
    )
    return _card_shell(
        f"📏 Document Chunking Minimap & Overlap Visualizer ({_esc(strategy_name)})",
        '<span style="color:#94a3b8;">● Dashed Vertical Lines = Markdown Headings</span> · <span style="color:#f59e0b;">█ Amber Highlights = Sliding Overlap Zones</span>',
        svg,
        margin="0 0 12px 0",
    )


def render_vector_hits_card(hits: Sequence[DuckDBChunkRecord], top_k: int) -> str:
    """Render ranked DuckDB vector search result bars inside a dark card."""
    rows = [
        _metric_row(
            f'<span style="color:#f8fafc;font-family:monospace;"><strong>#{idx + 1}</strong> · '
            f'<code>{_esc(r.doc_source)}</code> &gt; <strong>{_esc(r.section_title)}</strong> (Chunk #{r.chunk_id})</span>',
            f'<span style="color:#2dd4bf;font-family:monospace;font-weight:700;">vector_score = {r.score:.4f}</span>',
            max(5.0, min(100.0, r.score * 100.0)),
            "linear-gradient(90deg,#0284c7,#2dd4bf)",
            boxed=True,
            sub_html=f'<div style="font-size:12px;color:#94a3b8;font-family:ui-monospace,monospace;margin-top:5px;">'
            f'{_esc(_trunc(r.text.replace(chr(10), " "), 145, "..."))}</div>',
        )
        for idx, r in enumerate(hits)
    ]
    return _card_shell(
        f"📊 DuckDB Cosine Similarity Ranking (Top-{top_k} Results)",
        "",
        "".join(rows) or '<div style="color:#94a3b8;font-size:12px;">Enter a query to search DuckDB.</div>',
        margin="8px 0",
    )


def render_hybrid_rrf_svg(
    hybrid_results: list[HybridSearchResult],
    alpha: float,
    rrf_k: int,
) -> str:
    """Render stacked RRF score decomposition bars alongside the RRF decay curve."""
    if not hybrid_results:
        return ""

    max_rrf = max([1.0 / max(10, rrf_k)] + [r.rrf_score for r in hybrid_results])
    bars_svg: list[str] = []
    for idx, r in enumerate(hybrid_results):
        vec_c, bm25_c, vw, bw = _rrf_bar_parts(r, alpha, rrf_k, max_rrf, 185.0)
        bars_svg.append(
            _stacked_bar_row(
                y=36 + idx * 34,
                x_lbl=10,
                x_bar=135,
                bar_w=185,
                x_score=135 + int(vw + bw) + 6,
                title=f"#{r.final_rank} Chunk {r.chunk_id}",
                subtitle=f"V:#{r.vector_rank} B:#{r.bm25_rank}",
                sub_color="#94a3b8",
                vw=vw,
                bw=bw,
                score_txt=f"{r.rrf_score:.4f}",
                vec_tip=f"Vector RRF part: {vec_c:.5f} (Vector Rank #{r.vector_rank}, cos={r.vector_score:.3f})",
                bm25_tip=f"BM25 RRF part: {bm25_c:.5f} (BM25 Rank #{r.bm25_rank}, bm25={r.bm25_score:.2f})",
            )
        )

    curve_x0, curve_y0, curve_w, curve_h = 430, 42, 205, 175
    max_curve_y = 1.0 / 11.0

    def curve_points(k_val: int) -> str:
        return " ".join(
            f"{curve_x0 + ((rnk - 1) / 9.0) * curve_w:.1f},{(curve_y0 + curve_h) - ((1.0 / (k_val + rnk)) / max_curve_y) * curve_h:.1f}"
            for rnk in range(1, 11)
        )

    svg = _svg_frame(
        660,
        248,
        f"""
        <text x="10" y="22" fill="#cbd5e1" font-size="11" font-weight="bold" font-family="monospace">Stacked RRF Score per Retrieved Chunk</text>
        {"".join(bars_svg)}
        <line x1="395" y1="14" x2="395" y2="234" stroke="#1e293b" stroke-width="1"/>
        <text x="{curve_x0}" y="22" fill="#cbd5e1" font-size="11" font-weight="bold" font-family="monospace">RRF Rank Discount: 1 / (k + rank)</text>
        <line x1="{curve_x0}" y1="{curve_y0}" x2="{curve_x0}" y2="{curve_y0 + curve_h}" stroke="#334155" stroke-width="1"/>
        <line x1="{curve_x0}" y1="{curve_y0 + curve_h}" x2="{curve_x0 + curve_w}" y2="{curve_y0 + curve_h}" stroke="#334155" stroke-width="1"/>
        <text x="{curve_x0 + curve_w / 2:.1f}" y="{curve_y0 + curve_h + 20}" fill="#94a3b8" font-size="10" font-family="monospace" text-anchor="middle">Retriever Rank (1 → 10)</text>
        <polyline points="{curve_points(10)}" fill="none" stroke="#475569" stroke-width="1.2" stroke-dasharray="3 3"/>
        <text x="{curve_x0 + 32}" y="{curve_y0 + 18}" fill="#64748b" font-size="9.5" font-family="monospace">k=10 (steep)</text>
        <polyline points="{curve_points(60)}" fill="none" stroke="#475569" stroke-width="1.2" stroke-dasharray="3 3"/>
        <text x="{curve_x0 + curve_w - 5}" y="{curve_y0 + curve_h - 32}" fill="#64748b" font-size="9.5" font-family="monospace" text-anchor="end">k=60 (smooth)</text>
        <polyline points="{curve_points(rrf_k)}" fill="none" stroke="#a855f7" stroke-width="3"/>
        <text x="{curve_x0 + curve_w - 5}" y="{curve_y0 + 30}" fill="#e9d5ff" font-size="10.5" font-weight="bold" font-family="monospace" text-anchor="end">Active k={rrf_k}</text>
        """,
    )
    return _card_shell(
        f"⚖️ Interactive Reciprocal Rank Fusion (RRF) Decomposition (α = {alpha:.1f}, k = {rrf_k})",
        f'<span style="color:#2dd4bf;">█ Semantic Vector Contribution ({alpha:.0%})</span> · <span style="color:#f59e0b;">█ Lexical BM25 Contribution ({1.0 - alpha:.0%})</span>',
        svg,
    )


def render_rerank_slopegraph_svg(
    all_reranked: list[RerankedCandidate],
    top_k: int,
) -> str:
    """Render a Stage-1 -> Stage-2 Rank Slopegraph."""
    if not all_reranked:
        return ""

    row_h, width, x_left, x_right = 34, 660, 205, 445
    height = max(210, 56 + len(all_reranked) * row_h)
    cutoff_y = 40 + top_k * row_h
    lines_svg: list[str] = []
    nodes_svg: list[str] = []

    for item in all_reranked:
        y1 = 42 + (item.stage1_rank - 1) * row_h + 12
        y2 = 42 + (item.stage2_rank - 1) * row_h + 12
        if item.rank_delta > 0:
            col, delta_txt = "#2dd4bf", f"+{item.rank_delta} ↑"
        elif item.rank_delta < 0:
            col, delta_txt = "#f43f5e", f"{item.rank_delta} ↓"
        else:
            col, delta_txt = "#94a3b8", "0 ="

        in_top_k = item.stage2_rank <= top_k
        opacity = "0.95" if in_top_k or item.rank_delta != 0 else "0.45"
        stroke_w = "2.8" if in_top_k else "1.8"
        lines_svg.append(
            f'<path d="M {x_left} {y1} C {x_left + 85} {y1}, {x_right - 85} {y2}, {x_right} {y2}" '
            f'fill="none" stroke="{col}" stroke-width="{stroke_w}" opacity="{opacity}">'
            f'<title>Chunk #{item.chunk_id} ({_esc(item.section_title)}): Stage-1 #{item.stage1_rank} → Stage-2 #{item.stage2_rank} (CE={item.rerank_score:.4f})</title></path>'
        )
        short_sec = _trunc(item.section_title, 22)
        nodes_svg.append(
            f'<g><circle cx="{x_left}" cy="{y1}" r="5" fill="#64748b"/>'
            f'<text x="{x_left - 10}" y="{y1 + 4}" fill="#cbd5e1" font-size="10.5" font-family="monospace" text-anchor="end">'
            f'#{item.stage1_rank} Chunk {item.chunk_id} ({_esc(short_sec)})</text>'
            f'<circle cx="{x_right}" cy="{y2}" r="5.5" fill="{col}"/>'
            f'<text x="{x_right + 10}" y="{y2 + 4}" fill="{"#f8fafc" if in_top_k else "#64748b"}" '
            f'font-size="10.5" font-weight="{"bold" if in_top_k else "normal"}" font-family="monospace">'
            f'#{item.stage2_rank} Chunk {item.chunk_id} [CE={item.rerank_score:.3f} | {delta_txt}]</text></g>'
        )

    svg = _svg_frame(
        width,
        height,
        f"""
        <text x="{x_left - 10}" y="24" fill="#94a3b8" font-size="11" font-weight="bold" font-family="monospace" text-anchor="end">STAGE 1: Hybrid RRF Rank</text>
        <text x="{x_right + 10}" y="24" fill="#38bdf8" font-size="11" font-weight="bold" font-family="monospace">STAGE 2: Cross-Encoder Reranked</text>
        <rect x="{x_right - 12}" y="36" width="{width - x_right + 4}" height="{top_k * row_h}" rx="6" fill="#0f766e" opacity="0.16" stroke="#14b8a6" stroke-dasharray="3 3"/>
        <line x1="20" y1="{cutoff_y}" x2="{width - 15}" y2="{cutoff_y}" stroke="#f43f5e" stroke-width="1.2" stroke-dasharray="5 4"/>
        <text x="{(x_left + x_right) / 2:.1f}" y="{cutoff_y - 5}" fill="#fda4af" font-size="9.5" font-family="monospace" text-anchor="middle">✂️ Top-{top_k} Context Cutoff (Below this line is filtered out)</text>
        {"".join(lines_svg)}
        {"".join(nodes_svg)}
        """,
    )
    return _card_shell(
        "🔀 Stage-1 (Hybrid Recall) → Stage-2 (Cross-Encoder) Rank Slopegraph",
        f'<span style="color:#2dd4bf;">━ Promoted Direct Answers</span> · <span style="color:#f43f5e;">━ Demoted Distractors</span> · <span style="color:#38bdf8;">▭ Top-{top_k} Prompt Cutoff</span>',
        svg,
    )


def render_cross_attention_heatmap_svg(
    query: str,
    chunk_text: str,
    section_title: str,
    rerank_score: float,
    explanation: str,
) -> str:
    """Render a 2D Cross-Attention Token x Sentence interaction heatmap."""
    q_tokens = list(dict.fromkeys(tokenize(query, stem=False, drop_stopwords=True)))[:10] or ["query"]
    sentences = [s for s in split_sentences(chunk_text, min_len=9) if not s.startswith("#")][:5] or [chunk_text[:120]]

    col_w, row_h, pad_l, pad_t = 54, 28, 185, 52
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
        cells_svg.append(
            f'<text x="10" y="{y + 17}" fill="#cbd5e1" font-size="10" font-family="monospace">'
            f'S{r_idx + 1}: {_esc(_trunc(sent, 24, "..."))}<title>{_esc(sent)}</title></text>'
        )
        for c_idx, qt in enumerate(q_tokens):
            x = pad_l + c_idx * col_w
            exact_hit = qt in s_set
            partial_hit = (not exact_hit) and any(
                st.startswith(qt[:4]) or qt.startswith(st[:4])
                for st in s_set
                if len(st) >= 4 and len(qt) >= 4
            )
            fill, lbl, tip = (
                ("#14b8a6", "1.0", "Exact Match")
                if exact_hit
                else (("#0369a1", "0.5", "Stem Match") if partial_hit else ("#0f172a", "·", "No Match"))
            )
            cells_svg.append(
                f'<g><rect x="{x + 2}" y="{y + 2}" width="{col_w - 4}" height="{row_h - 4}" rx="4" fill="{fill}" stroke="#1e293b">'
                f'<title>Query token "{_esc(qt)}" × Sentence {r_idx + 1}: {tip}</title></rect>'
                f'<text x="{x + col_w / 2:.1f}" y="{y + 17}" fill="{"#020617" if exact_hit else "#64748b"}" font-size="10" font-weight="bold" font-family="monospace" text-anchor="middle">{lbl}</text></g>'
            )

    svg = _svg_frame(
        width,
        height,
        f'<text x="10" y="22" fill="#94a3b8" font-size="10" font-family="monospace">Passage Sentences ↓ \\ Query Tokens →</text>'
        f'{"".join(headers_svg)}{"".join(cells_svg)}',
        responsive=False,
    )
    return _card_shell(
        f"🎯 Cross-Encoder Token × Sentence Attention Matrix: <code>{_esc(section_title)}</code>",
        f'<span style="font-family:monospace;color:#2dd4bf;">Score = {rerank_score:.4f} ({_esc(explanation)})</span>',
        svg,
        margin="10px 0 0 0",
    )


def build_agentic_mermaid_diagram(trace: AgenticTrace) -> str:
    """Build a dynamic Mermaid flowchart for a live AgenticTrace."""
    lines = [
        "flowchart LR",
        f'    Q["User Query: {_clean_mermaid_label(trace.original_query, 36)}"] --> Plan{{"Query Planner ({("Multi-Hop" if trace.is_multi_hop else "Single-Hop")})"}}',
    ]
    for step in trace.retrieval_steps:
        sq_id, ch_id = f"SQ{step.step_number}", f"CH{step.step_number}"
        ch_lbl = f"Chunk #{step.top_chunk_id}: {_clean_mermaid_label(step.top_section, 22)} ({step.rerank_score:.2f})"
        lines.append(f'    Plan --> {sq_id}["Step {step.step_number}: {_clean_mermaid_label(step.sub_query, 32)}"]')
        lines.append(f'    {sq_id} --> {ch_id}[("{ch_lbl}")]')
        lines.append(f"    {ch_id} --> Gate")
    gate_status = "Pass" if trace.sufficiency_passed else "Retry Triggered"
    lines.append(f'    Gate{{"Sufficiency Gate: {gate_status}"}} --> LLM["Llama.app Synthesis ({len(trace.final_context_chunks)} Chunks)"]')
    lines.append(f'    LLM --> Audit["Faithfulness Audit: {trace.faithfulness_score:.0%}"]')
    return "\n".join(lines)


def render_faithfulness_audit_svg(trace: AgenticTrace) -> str:
    """Render visual confidence bars for audited claims in an AgenticTrace."""
    if not trace.verifications:
        return ""
    rows_html = []
    for idx, v in enumerate(trace.verifications):
        badge = (
            f'<span style="background:#064e3b;color:#6ee7b7;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:11px;">'
            f"✅ Cited Chunk #{v.supporting_chunk_id} ({v.confidence:.2f})</span>"
            if v.is_grounded
            else f'<span style="background:#4c0519;color:#fda4af;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:11px;">'
            f"⚠️ Unverified ({v.confidence:.2f})</span>"
        )
        rows_html.append(_metric_row(
            f'<span style="color:#e2e8f0;flex:1;"><strong>Claim #{idx + 1}:</strong> {_esc(v.claim)}</span>',
            badge,
            max(4.0, min(100.0, v.confidence * 100.0)),
            "#2dd4bf" if v.is_grounded else "#f43f5e",
            bar_h=7, marker_pct=35.0, marker_title="Verification Threshold (0.35)", boxed=True,
        ))
    return _card_shell(
        f"🛡️ Claim-by-Claim Grounding & Faithfulness Meter (Overall: {trace.faithfulness_score:.0%})",
        '<span style="color:#fbbf24;font-family:monospace;">│ Amber Line = 0.35 Grounding Threshold</span>',
        "".join(rows_html),
        margin="8px 0 0 0",
    )


def build_studio_pipeline_mermaid(
    chunk_strategy: str,
    alpha: float,
    use_reranker: bool,
    use_agentic: bool,
    top_k: int,
    total_chunks: int,
) -> str:
    """Build a dynamic Mermaid flowchart for Notebook 4 Pipeline B."""
    chunk_lbl = "Markdown Section Chunker" if chunk_strategy == "markdown" else "Fixed-Char Sliding Window"
    hybrid_lbl = f"Hybrid RRF (α={alpha:.1f} Vec / {1.0 - alpha:.1f} BM25)"
    rerank_node = (
        f'Rerank["Stage-2 Cross-Encoder (Top-{top_k})"]'
        if use_reranker or use_agentic
        else f'Rerank["Reranker Bypassed (Direct Top-{top_k})"]'
    )
    agent_node = 'Plan{{"Agentic Multi-Hop Planner"}}' if use_agentic else 'Plan["Single-Shot Query"]'
    verify_node = 'Verify["Faithfulness Audit & Citations"]' if use_agentic else 'Verify["Grounded Answer"]'
    return "\n".join([
        "flowchart LR",
        f'    Docs["Corpus + Custom Doc"] --> Chunker["{chunk_lbl}"]',
        f'    Chunker --> DB[("DuckDB Store ({total_chunks} Chunks)")]',
        f'    Q["User Question"] --> {agent_node}',
        f'    Plan --> Hybrid["{hybrid_lbl}"]',
        "    DB --> Hybrid",
        f"    Hybrid --> {rerank_node}",
        '    Rerank --> LLM["Llama.app Synthesis"]',
        f"    LLM --> {verify_node}",
    ])


def render_ab_arena_comparison_svg(
    baseline_hits: list[tuple[str, float]],
    prod_chunks: list[tuple[str, float, str]],
) -> str:
    """Render side-by-side relevance bars for Pipeline A vs. Pipeline B."""

    def _bar_rows(items: list[tuple[str, float, str]], val_color: str, bar_fill: str) -> str:
        return "".join(
            _metric_row(
                f'<span style="color:#e2e8f0;font-family:monospace;"><strong>#{i + 1}</strong> {_esc(lbl)}</span>',
                f'<span style="color:{val_color};font-family:monospace;font-weight:700;">{sc_lbl}</span>',
                pct, bar_fill,
            )
            for i, (lbl, pct, sc_lbl) in enumerate(items)
        )

    items_a = [
        (_trunc(txt.replace("\n", " "), 68, "..."), max(5.0, min(100.0, sc * 100.0)), f"cos={sc:.3f}")
        for txt, sc in baseline_hits
    ]
    items_b = [
        (lbl[:54], max(8.0, min(100.0, (sc * 100.0) if sc > 0.05 else (sc * 3200.0))), f"score={sc:.3f}")
        for lbl, sc, _ in prod_chunks
    ]
    return f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;background:#0f172a;padding:12px 14px;border-radius:12px;border:1px solid #334155;color:#f8fafc;margin:10px 0;">
      <div style="background:#020617;padding:10px 12px;border-radius:8px;border:1px solid #1e293b;">
        <div style="font-size:12.5px;font-weight:700;color:#38bdf8;margin-bottom:8px;">📊 Pipeline A: Scratch Vector Retrieval Scores</div>
        {_bar_rows(items_a, "#38bdf8", "#38bdf8")}
      </div>
      <div style="background:#020617;padding:10px 12px;border-radius:8px;border:1px solid #1e293b;">
        <div style="font-size:12.5px;font-weight:700;color:#2dd4bf;margin-bottom:8px;">🚀 Pipeline B: Production Retrieval Scores</div>
        {_bar_rows(items_b, "#2dd4bf", "linear-gradient(90deg,#14b8a6,#2dd4bf)")}
      </div>
    </div>
    """


def render_subgraph_traversal_svg(graph_result: GraphSearchResult) -> str:
    """Render a layered 3-column SVG showing Seed -> 1-Hop -> 2-Hop graph traversal."""
    triples = graph_result.matched_triples[:8]
    if not triples:
        return (
            '<div style="background:#0f172a;padding:14px;border-radius:12px;border:1px solid #334155;color:#94a3b8;font-size:13px;">'
            "🕸️ <strong>No Knowledge Graph entities matched this query.</strong> Try including terms like "
            "<code>DuckDB</code>, <code>RRF-K60</code>, <code>Cross-Encoder</code>, <code>HyDE</code>, or <code>Qdrant</code>.</div>"
        )

    hop_level: dict[str, int] = {s: 0 for s in graph_result.seed_entities}
    for triple, hop in triples:
        for ent in (triple.subject, triple.obj):
            hop_level.setdefault(ent, min(2, hop))

    cols: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for ent, lvl in hop_level.items():
        cols[min(2, lvl)].append(ent)

    max_rows = max(len(cols[0]), len(cols[1]), len(cols[2]), 1)
    row_h, width = 46, 720
    height = max(210, 62 + max_rows * row_h)
    col_x = {0: 105, 1: 360, 2: 615}
    col_colors = {
        0: ("#0c4a6e", "#38bdf8", "Seed Match"),
        1: ("#134e4a", "#2dd4bf", "1-Hop Entity"),
        2: ("#3b0764", "#c084fc", "2-Hop Expansion"),
    }

    positions: dict[str, tuple[float, float, int]] = {}
    for lvl, ents in cols.items():
        top_offset = 46 + ((max_rows - len(ents)) * row_h) / 2.0
        for idx, ent in enumerate(ents):
            positions[ent] = (col_x[lvl], top_offset + idx * row_h + 14, lvl)

    edges_svg: list[str] = []
    for t_idx, (triple, hop) in enumerate(triples):
        if triple.subject not in positions or triple.obj not in positions:
            continue
        x1, y1, _ = positions[triple.subject]
        x2, y2, _ = positions[triple.obj]
        edge_col = "#2dd4bf" if hop == 1 else "#a855f7"
        if x1 == x2:
            ctrl_x = x1 + 92
            path_d = f"M {x1 + 75:.1f} {y1:.1f} Q {ctrl_x:.1f} {(y1 + y2) / 2:.1f}, {x2 + 75:.1f} {y2:.1f}"
            mid_x, mid_y = ctrl_x - 8, (y1 + y2) / 2.0
        else:
            sx, ex = (x1 + 78, x2 - 78) if x1 < x2 else (x1 - 78, x2 + 78)
            path_d = f"M {sx:.1f} {y1:.1f} L {ex:.1f} {y2:.1f}"
            mid_x, mid_y = (sx + ex) / 2.0, (y1 + y2) / 2.0 + (-7 if t_idx % 2 == 0 else 9)
        edges_svg.append(
            f'<g><path d="{path_d}" fill="none" stroke="{edge_col}" stroke-width="1.8" opacity="0.85">'
            f'<title>{_esc(triple.subject)} --[{_esc(triple.predicate)}]--> {_esc(triple.obj)} (Chunk #{triple.source_chunk_id})</title></path>'
            f'<text x="{mid_x:.1f}" y="{mid_y:.1f}" fill="#cbd5e1" font-size="9" font-family="monospace" text-anchor="middle">{_esc(triple.predicate[:18])}</text></g>'
        )

    nodes_svg = [
        f'<g><rect x="{nx - 78:.1f}" y="{ny - 15:.1f}" width="156" height="30" rx="7" fill="{col_colors[lvl][0]}" stroke="{col_colors[lvl][1]}" stroke-width="1.6">'
        f'<title>{_esc(ent)} ({col_colors[lvl][2]})</title></rect>'
        f'<text x="{nx:.1f}" y="{ny + 4:.1f}" fill="#f8fafc" font-size="10.5" font-weight="bold" font-family="monospace" text-anchor="middle">'
        f'{_esc(_trunc(ent, 20))}</text></g>'
        for ent, (nx, ny, lvl) in positions.items()
    ]

    svg = _svg_frame(
        width,
        height,
        f'<text x="{col_x[0]}" y="24" fill="#38bdf8" font-size="10.5" font-weight="bold" font-family="monospace" text-anchor="middle">QUERY SEEDS</text>'
        f'<text x="{col_x[1]}" y="24" fill="#2dd4bf" font-size="10.5" font-weight="bold" font-family="monospace" text-anchor="middle">1-HOP NEIGHBORS</text>'
        f'<text x="{col_x[2]}" y="24" fill="#c084fc" font-size="10.5" font-weight="bold" font-family="monospace" text-anchor="middle">2-HOP EXPANSION</text>'
        f'{"".join(edges_svg)}{"".join(nodes_svg)}',
    )
    return _card_shell(
        f"🕸️ Knowledge Graph 1–2 Hop Subgraph Traversal ({len(triples)} Triples Matched)",
        '<span style="color:#38bdf8;">■ Seed Entities</span> · <span style="color:#2dd4bf;">■ 1-Hop Neighbors</span> · <span style="color:#c084fc;">■ 2-Hop Expansion</span>',
        svg,
    )


def render_fusion_comparison_svg(
    rrf_results: list[HybridSearchResult],
    convex_results: list[HybridSearchResult],
    alpha: float,
    rrf_k: int,
) -> str:
    """Render side-by-side RRF vs. Min-Max Convex Combination score breakdown bars."""
    if not rrf_results or not convex_results:
        return ""

    top_n = min(5, len(rrf_results), len(convex_results))
    rrf_slice, conv_slice = rrf_results[:top_n], convex_results[:top_n]
    rrf_rank_map = {r.chunk_id: r.final_rank for r in rrf_results}
    max_rrf = max([1.0 / max(10, rrf_k)] + [r.rrf_score for r in rrf_slice])
    row_h, width = 34, 680
    height = 56 + top_n * row_h

    rrf_bars = [
        _stacked_bar_row(
            y=38 + idx * row_h,
            x_lbl=12,
            x_bar=108,
            bar_w=145,
            x_score=260,
            title=f"#{r.final_rank} Chunk {r.chunk_id}",
            subtitle=f"V#{r.vector_rank} / B#{r.bm25_rank}",
            sub_color="#64748b",
            vw=vw,
            bw=bw,
            score_txt=f"{r.rrf_score:.4f}",
        )
        for idx, r in enumerate(rrf_slice)
        for _, _, vw, bw in (_rrf_bar_parts(r, alpha, rrf_k, max_rrf, 145.0),)
    ]

    conv_bars: list[str] = []
    for idx, c in enumerate(conv_slice):
        vw = min(145.0, alpha * c.norm_vector_score * 145.0)
        bw = min(145.0 - vw, (1.0 - alpha) * c.norm_bm25_score * 145.0)
        shift = rrf_rank_map.get(c.chunk_id, c.final_rank) - c.final_rank
        shift_txt = f"↑+{shift}" if shift > 0 else (f"↓{shift}" if shift < 0 else "=")
        shift_col = "#2dd4bf" if shift > 0 else ("#f43f5e" if shift < 0 else "#64748b")
        conv_bars.append(
            _stacked_bar_row(
                y=38 + idx * row_h,
                x_lbl=355,
                x_bar=452,
                bar_w=145,
                x_score=604,
                title=f"#{c.final_rank} Chunk {c.chunk_id}",
                subtitle=f"vs RRF: {shift_txt}",
                sub_color=shift_col,
                vw=vw,
                bw=bw,
                score_txt=f"{c.convex_score:.3f}",
            )
        )

    svg = _svg_frame(
        width,
        height,
        f'<text x="12" y="22" fill="#38bdf8" font-size="11" font-weight="bold" font-family="monospace">Reciprocal Rank Fusion (Rank-Based)</text>'
        f'<text x="355" y="22" fill="#c084fc" font-size="11" font-weight="bold" font-family="monospace">Convex Linear Blend (Min-Max [0..1])</text>'
        f'<line x1="340" y1="12" x2="340" y2="{height - 12}" stroke="#1e293b" stroke-width="1"/>'
        f'{"".join(rrf_bars)}{"".join(conv_bars)}',
    )
    return _card_shell(
        f"⚖️ Score Fusion Arena: Reciprocal Rank Fusion (k={rrf_k}) vs. Convex Combination (α={alpha:.1f})",
        f'<span style="color:#2dd4bf;">█ Dense Vector ({alpha:.0%})</span> · <span style="color:#f59e0b;">█ Sparse BM25 ({1.0 - alpha:.0%})</span>',
        svg,
    )


def build_modular_rag_mermaid(
    config: ModularRAGConfig,
    expansion: QueryExpansionResult | None = None,
    graph_result: GraphSearchResult | None = None,
) -> str:
    """Build a dynamic Mermaid diagram showing active Modular RAG stages."""
    if config.expansion_mode == "multi_query":
        n_q = len(expansion.expanded_queries) if expansion else 3
        pre_node = f'Pre["Multi-Query Expansion ({n_q} variants)"]'
    elif config.expansion_mode == "hyde":
        pre_node = 'Pre["HyDE: Hypothetical Answer Embedding"]'
    else:
        pre_node = 'Pre["Direct Query (No Expansion)"]'

    fusion_lbl = (
        f"RRF Fusion (k={config.rrf_k}, α={config.alpha:.1f})"
        if config.fusion_method == "rrf"
        else f"Convex Blend (α={config.alpha:.1f} Vec + {1.0 - config.alpha:.1f} BM25)"
    )

    lines = [
        "flowchart LR",
        '    Q["User Query"] --> ' + pre_node,
    ]
    if config.enable_dense_sparse:
        since_lbl = _clean_mermaid_label(str(config.min_year) if config.min_year is not None else "any", 12)
        sql_note = (
            f" + SQL Filter ({_clean_mermaid_label(config.department or 'all', 18)}, since {since_lbl})"
            if (config.department and config.department != "all") or config.min_year is not None
            else ""
        )
        lines.append(f'    Pre --> Dense["Dense Vector Search (DuckDB 128-D){sql_note}"]')
        lines.append('    Pre --> Sparse["Sparse Keyword Search (Okapi BM25)"]')
        lines.append(f'    Dense --> Fuse{{"{fusion_lbl}"}}')
        lines.append("    Sparse --> Fuse")
    else:
        lines.append(f'    Pre --> Fuse{{"{fusion_lbl}"}}')

    if config.enable_graph:
        n_triples = len(graph_result.matched_triples) if graph_result else 0
        lines.append(f'    Pre --> Graph[("Knowledge Graph ({n_triples} triples)")]')
        lines.append("    Graph --> Fuse")

    if config.use_reranker:
        lines.append(f'    Fuse --> Rerank["Stage-2 Cross-Encoder (Top-{config.stage1_k} -> Top-{config.final_top_k})"]')
    else:
        lines.append(f'    Fuse --> Rerank["Reranker Bypassed (Top-{config.final_top_k})"]')

    lines.append('    Rerank --> LLM["Grounded LLM Synthesis (Llama.app)"]')
    return "\n".join(lines)
