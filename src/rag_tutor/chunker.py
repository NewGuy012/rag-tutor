"""Document Chunking Strategies & Corpus Loader."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")

_DOC_METADATA_DEFAULTS: dict[str, tuple[str, str, int]] = {
    "01_rag_fundamentals.md": ("foundations", "core-curriculum", 2024),
    "02_duckdb_vectors.md": ("storage-engineering", "data-platform", 2025),
    "03_production_rag_patterns.md": ("ml-production", "retrieval-team", 2025),
    "04_advanced_graph_modular_rag.md": ("advanced-architecture", "graph-modular-team", 2026),
    "custom_runbook.md": ("operations", "oncall-sre", 2026),
}


def infer_doc_metadata(doc_source: str) -> tuple[str, str, int]:
    """Return `(department, author, updated_year)` metadata for a document source."""
    return _DOC_METADATA_DEFAULTS.get(
        Path(doc_source).name,
        ("general", "rag-tutor", 2026),
    )


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
    author: str = "rag-tutor"
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
        m = _HEADING_RE.match(line.strip())
        if m:
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

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 1
    n = len(cleaned)
    department, author, updated_year = infer_doc_metadata(doc_source)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            ws_pos = cleaned.rfind(" ", start + int(chunk_size * 0.6), end)
            if ws_pos != -1:
                end = ws_pos

        snippet = cleaned[start:end].strip()
        if len(snippet) >= 15:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_source=doc_source,
                    section_title=f"Window #{chunk_id} ({start}..{end})",
                    text=snippet,
                    start_char=start,
                    end_char=end,
                    overlap_chars=chunk_overlap if chunk_id > 1 else 0,
                    department=department,
                    author=author,
                    updated_year=updated_year,
                )
            )
            chunk_id += 1

        if end >= n:
            break
        start = max(start + 1, end - chunk_overlap)

    return chunks


def chunk_markdown_sections(
    text: str,
    max_chunk_size: int = 550,
    chunk_overlap: int = 80,
    doc_source: str = "document.md",
) -> list[Chunk]:
    """Split Markdown by headings (`#`, `##`, `###`) and subdivide long sections."""
    sections: list[tuple[str, str, int, int]] = []
    current_title = "Introduction"
    buffer: list[str] = []
    sec_start = 0
    cursor = 0

    for line in text.splitlines(keepends=True):
        heading_match = _HEADING_RE.match(line.strip())
        if heading_match and buffer:
            sec_text = "".join(buffer).strip()
            if len(sec_text) >= 15:
                sections.append((current_title, sec_text, sec_start, cursor))
            current_title = heading_match.group(2).strip()
            buffer = [line]
            sec_start = cursor
        else:
            if heading_match:
                current_title = heading_match.group(2).strip()
            buffer.append(line)
        cursor += len(line)

    if buffer:
        sec_text = "".join(buffer).strip()
        if len(sec_text) >= 15:
            sections.append((current_title, sec_text, sec_start, cursor))

    chunks: list[Chunk] = []
    chunk_id = 1
    department, author, updated_year = infer_doc_metadata(doc_source)
    for sec_title, sec_text, s_start, s_end in sections:
        if len(sec_text) <= max_chunk_size:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_source=doc_source,
                    section_title=sec_title,
                    text=sec_text,
                    start_char=s_start,
                    end_char=s_end,
                    overlap_chars=0,
                    department=department,
                    author=author,
                    updated_year=updated_year,
                )
            )
            chunk_id += 1
        else:
            sub_chunks = chunk_fixed_character(
                sec_text,
                chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
                doc_source=doc_source,
            )
            for idx, sc in enumerate(sub_chunks):
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_source=doc_source,
                        section_title=f"{sec_title} (Part {idx + 1})",
                        text=sc.text,
                        start_char=s_start + sc.start_char,
                        end_char=s_start + sc.end_char,
                        overlap_chars=sc.overlap_chars,
                        department=department,
                        author=author,
                        updated_year=updated_year,
                    )
                )
                chunk_id += 1

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
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_source=doc_source,
        )
    return chunk_markdown_sections(
        text,
        max_chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        doc_source=doc_source,
    )


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
        doc_chunks = chunk_document(
            md_path.read_text(encoding="utf-8"),
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            doc_source=md_path.name,
        )
        for ch in doc_chunks:
            ch.chunk_id = len(all_chunks) + 1
            all_chunks.append(ch)
    return all_chunks
