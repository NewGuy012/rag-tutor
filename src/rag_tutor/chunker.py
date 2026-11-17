"""Document Chunking Strategies & Corpus Loader."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MIN_CHUNK_CHARS = 15
_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")

_DOC_METADATA_DEFAULTS: dict[str, tuple[str, int]] = {
    "01_rag_fundamentals.md": ("foundations", 2024),
    "02_duckdb_vectors.md": ("storage-engineering", 2025),
    "03_production_rag_patterns.md": ("ml-production", 2025),
    "04_advanced_graph_modular_rag.md": ("advanced-architecture", 2026),
    "custom_runbook.md": ("operations", 2026),
}


def infer_doc_metadata(doc_source: str) -> tuple[str, int]:
    """Return `(department, updated_year)` metadata for a document source."""
    return _DOC_METADATA_DEFAULTS.get(Path(doc_source).name, ("general", 2026))


@dataclass
class Chunk:
    """A single document chunk with provenance and boundary metadata."""

    chunk_id: int
    doc_source: str
    section_title: str
    text: str
    start_char: int
    end_char: int
    overlap_chars: int = 0
    department: str = "general"
    updated_year: int = 2026

    @property
    def char_length(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def extract_markdown_headings(text: str) -> list[tuple[int, str]]:
    """Return `(char_offset, heading_title)` for all `#`..`###` headings in `text`."""
    headings: list[tuple[int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        if m := _HEADING_RE.match(line.strip()):
            headings.append((cursor, m.group(2).strip()))
        cursor += len(line)
    return headings


def chunk_fixed_character(
    text: str,
    chunk_size: int = 320,
    chunk_overlap: int = 64,
    doc_source: str = "document.md",
) -> list[Chunk]:
    """Split text into fixed-character windows with configurable sliding overlap."""
    chunk_size = max(60, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 20))

    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start, n = 0, len(text)
    department, updated_year = infer_doc_metadata(doc_source)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            ws_pos = text.rfind(" ", start + int(chunk_size * 0.6), end)
            if ws_pos != -1:
                end = ws_pos

        raw_slice = text[start:end]
        snippet = raw_slice.strip()
        if len(snippet) >= MIN_CHUNK_CHARS:
            lead = len(raw_slice) - len(raw_slice.lstrip())
            c_start = start + lead
            c_end = c_start + len(snippet)
            cid = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_source=doc_source,
                    section_title=f"Window #{cid} ({c_start}..{c_end})",
                    text=snippet,
                    start_char=c_start,
                    end_char=c_end,
                    overlap_chars=max(0, chunks[-1].end_char - c_start) if chunks else 0,
                    department=department,
                    updated_year=updated_year,
                )
            )

        if end >= n:
            break
        start = end if not snippet else max(start + 1, end - chunk_overlap)

    return chunks


def chunk_markdown_sections(
    text: str,
    max_chunk_size: int = 550,
    chunk_overlap: int = 80,
    doc_source: str = "document.md",
) -> list[Chunk]:
    """Split Markdown by `#`..`###` headings and subdivide long sections."""
    sections: list[tuple[str, str, int, int]] = []
    title, buf, sec_start, cursor = "Introduction", [], 0, 0
    for line in [*text.splitlines(keepends=True), None]:
        m = _HEADING_RE.match(line.strip()) if line is not None else None
        if buf and (line is None or m):
            raw_sec = "".join(buf)
            sec_text = raw_sec.strip()
            if len(sec_text) >= MIN_CHUNK_CHARS:
                lead = len(raw_sec) - len(raw_sec.lstrip())
                c_start = sec_start + lead
                sections.append((title, sec_text, c_start, c_start + len(sec_text)))
            buf, sec_start = [], cursor
        if line is not None:
            if m:
                title = m.group(2).strip()
            buf.append(line)
            cursor += len(line)

    chunks: list[Chunk] = []
    department, updated_year = infer_doc_metadata(doc_source)
    for sec_title, sec_text, s_start, s_end in sections:
        if len(sec_text) <= max_chunk_size:
            chunks.append(
                Chunk(
                    chunk_id=len(chunks) + 1,
                    doc_source=doc_source,
                    section_title=sec_title,
                    text=sec_text,
                    start_char=s_start,
                    end_char=s_end,
                    department=department,
                    updated_year=updated_year,
                )
            )
        else:
            for idx, sc in enumerate(
                chunk_fixed_character(sec_text, max_chunk_size, chunk_overlap, doc_source)
            ):
                sc.chunk_id = len(chunks) + 1
                sc.section_title = f"{sec_title} (Part {idx + 1})"
                sc.start_char += s_start
                sc.end_char += s_start
                chunks.append(sc)

    return chunks


def chunk_document(
    text: str,
    strategy: str = "markdown",
    chunk_size: int = 480,
    chunk_overlap: int = 60,
    doc_source: str = "document.md",
) -> list[Chunk]:
    """Dispatch document chunking by strategy (`'markdown'` or `'fixed'`)."""
    if strategy == "fixed":
        return chunk_fixed_character(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, doc_source=doc_source
        )
    if strategy == "markdown":
        return chunk_markdown_sections(
            text, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap, doc_source=doc_source
        )
    raise ValueError(f"Unsupported chunking strategy {strategy!r}; expected 'markdown' or 'fixed'.")


def load_sample_markdown_corpus(
    corpus_dir: str | Path | None = None,
    strategy: str = "markdown",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """Load and chunk all `.md` files in `data/sample_docs/`."""
    corpus_path = (
        Path(__file__).resolve().parents[2] / "data" / "sample_docs"
        if corpus_dir is None
        else Path(corpus_dir)
    )

    all_chunks: list[Chunk] = []
    for md_path in sorted(corpus_path.glob("*.md")):
        for ch in chunk_document(
            md_path.read_text(encoding="utf-8"),
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_source=md_path.name,
        ):
            ch.chunk_id = len(all_chunks) + 1
            all_chunks.append(ch)
    return all_chunks
