"""Module 4: Document Chunking Strategies & Overlap Visualization Helpers.

Provides line-based, fixed-character with sliding overlap, and Markdown
section-aware chunking strategies so learners can explore how chunk size and
overlap affect retrieval quality.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


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

    @property
    def char_length(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def chunk_by_lines(text: str, doc_source: str = "ev-facts.txt") -> list[Chunk]:
    """Split text one chunk per non-empty line (ideal for 1-fact-per-line datasets)."""
    chunks: list[Chunk] = []
    cursor = 0
    chunk_id = 1
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        start = cursor
        end = cursor + len(line)
        cursor = end
        if len(stripped) >= 10:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_source=doc_source,
                    section_title=f"Line #{chunk_id}",
                    text=stripped,
                    start_char=start,
                    end_char=end,
                    overlap_chars=0,
                )
            )
            chunk_id += 1
    return chunks


def chunk_fixed_character(
    text: str,
    chunk_size: int = 320,
    chunk_overlap: int = 64,
    doc_source: str = "document.md",
) -> list[Chunk]:
    """Split text into fixed-character windows with configurable sliding overlap."""
    chunk_size = max(60, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 20))
    step = chunk_size - chunk_overlap

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_id = 1
    n = len(cleaned)

    while start < n:
        end = min(start + chunk_size, n)
        # Prefer breaking on whitespace if we are not at the end of the text
        if end < n:
            ws_pos = cleaned.rfind(" ", start + int(chunk_size * 0.6), end)
            if ws_pos != -1:
                end = ws_pos

        snippet = cleaned[start:end].strip()
        if len(snippet) >= 15:
            actual_overlap = chunk_overlap if chunk_id > 1 else 0
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_source=doc_source,
                    section_title=f"Window #{chunk_id} ({start}..{end})",
                    text=snippet,
                    start_char=start,
                    end_char=end,
                    overlap_chars=actual_overlap,
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
    lines = text.splitlines(keepends=True)
    sections: list[tuple[str, str, int, int]] = []

    current_title = "Introduction"
    buffer: list[str] = []
    sec_start = 0
    cursor = 0

    for line in lines:
        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line.strip())
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
                    )
                )
                chunk_id += 1

    return chunks


def load_sample_markdown_corpus(
    corpus_dir: str | Path | None = None,
    strategy: str = "markdown",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[Chunk]:
    """Load and chunk all `.md` files in `data/sample_docs/`."""
    if corpus_dir is None:
        corpus_dir = Path(__file__).resolve().parents[2] / "data" / "sample_docs"
    else:
        corpus_dir = Path(corpus_dir)

    all_chunks: list[Chunk] = []
    next_id = 1
    for md_path in sorted(corpus_dir.glob("*.md")):
        raw = md_path.read_text(encoding="utf-8")
        if strategy == "fixed":
            doc_chunks = chunk_fixed_character(
                raw,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_source=md_path.name,
            )
        else:
            doc_chunks = chunk_markdown_sections(
                raw,
                max_chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                doc_source=md_path.name,
            )
        for ch in doc_chunks:
            ch.chunk_id = next_id
            next_id += 1
            all_chunks.append(ch)
    return all_chunks
