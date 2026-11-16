"""Module 5: DuckDB Embedded Vector & Full-Text Search Store.

Demonstrates how DuckDB stores document chunks, metadata, and fixed-size
`FLOAT[N]` embedding arrays, executing native `array_cosine_similarity` queries
in SQL as well as Okapi BM25 lexical scoring.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

try:
    import duckdb as _duckdb_mod
except ImportError:
    _duckdb_mod = None

from .chunker import Chunk
from .llm_backend import get_default_embedder
from .scratch_rag import cosine_similarity


@dataclass
class DuckDBChunkRecord:
    """A scored chunk record returned from DuckDB."""

    chunk_id: int
    doc_source: str
    section_title: str
    text: str
    score: float


class DuckDBVectorStore:
    """Embedded DuckDB vector and BM25 store for RAG pipelines.

    Uses the `duckdb` Python package when running inside the `uv` environment,
    and also supports the `/opt/homebrew/bin/duckdb` CLI so SQL queries and
    vector searches work in every environment.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_dim: int = 128,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.embed_fn = embed_fn or get_default_embedder().embed
        self._mem_rows: list[tuple[int, str, str, str, int, list[float]]] = []
        self.conn = _duckdb_mod.connect(db_path) if _duckdb_mod is not None else None
        self._duckdb_cli = shutil.which("duckdb") or "/opt/homebrew/bin/duckdb"
        self._init_schema()

    def _init_schema(self) -> None:
        """Create the `rag_chunks` table with a fixed-size `FLOAT[dim]` column."""
        if self.conn is not None:
            self.conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id INTEGER PRIMARY KEY,
                    doc_source VARCHAR,
                    section_title VARCHAR,
                    chunk_text VARCHAR,
                    char_length INTEGER,
                    embedding FLOAT[{self.embedding_dim}]
                );
                """
            )

    def clear(self) -> None:
        """Remove all rows from `rag_chunks`."""
        self._mem_rows.clear()
        if self.conn is not None:
            self.conn.execute("DELETE FROM rag_chunks;")

    def index_chunks(self, chunks: list[Chunk], replace: bool = True) -> int:
        """Embed and insert a list of `Chunk` objects into DuckDB."""
        if replace:
            self.clear()

        rows: list[tuple[int, str, str, str, int, list[float]]] = []
        for ch in chunks:
            cleaned = ch.text.strip()
            if len(cleaned) < 10:
                continue
            vec = self.embed_fn(cleaned)
            if len(vec) != self.embedding_dim:
                raise ValueError(
                    f"ERR-VEC-768: Expected {self.embedding_dim}-D embedding, got {len(vec)}-D"
                )
            rows.append(
                (
                    ch.chunk_id,
                    ch.doc_source,
                    ch.section_title,
                    cleaned,
                    len(cleaned),
                    vec,
                )
            )

        self._mem_rows.extend(rows)
        if self.conn is not None and rows:
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO rag_chunks
                (chunk_id, doc_source, section_title, chunk_text, char_length, embedding)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
        return self.count()

    def count(self) -> int:
        """Return total number of indexed chunks in DuckDB."""
        if self.conn is not None:
            row = self.conn.execute("SELECT COUNT(*) FROM rag_chunks;").fetchone()
            return int(row[0]) if row else 0
        return len(self._mem_rows)

    def search_vector(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[DuckDBChunkRecord]:
        """Run exact cosine similarity search in DuckDB (`array_cosine_similarity`)."""
        query_vec = self.embed_fn(query)
        if self.conn is not None:
            sql = f"""
                SELECT
                    chunk_id,
                    doc_source,
                    section_title,
                    chunk_text,
                    array_cosine_similarity(
                        embedding,
                        ?::FLOAT[{self.embedding_dim}]
                    ) AS vector_score
                FROM rag_chunks
                ORDER BY vector_score DESC NULLS LAST
                LIMIT ?;
            """
            rows = self.conn.execute(sql, [query_vec, int(top_k)]).fetchall()
            return [
                DuckDBChunkRecord(
                    chunk_id=int(r[0]),
                    doc_source=str(r[1]),
                    section_title=str(r[2]),
                    text=str(r[3]),
                    score=float(r[4] or 0.0),
                )
                for r in rows
            ]

        scored = [
            DuckDBChunkRecord(
                chunk_id=cid,
                doc_source=src,
                section_title=sec,
                text=txt,
                score=round(cosine_similarity(query_vec, vec), 5),
            )
            for cid, src, sec, txt, _, vec in self._mem_rows
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[DuckDBChunkRecord]:
        """Compute Okapi BM25 (`BM25-K1-1.2`, `b=0.75`) over chunks stored in DuckDB."""
        if self.conn is not None:
            rows = self.conn.execute(
                """
                SELECT chunk_id, doc_source, section_title, chunk_text
                FROM rag_chunks
                ORDER BY chunk_id ASC;
                """
            ).fetchall()
        else:
            rows = [(cid, src, sec, txt) for cid, src, sec, txt, _, _ in self._mem_rows]

        if not rows:
            return []

        def tokenize_bm25(text: str) -> list[str]:
            return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())

        query_terms = tokenize_bm25(query)
        if not query_terms:
            return [
                DuckDBChunkRecord(int(r[0]), str(r[1]), str(r[2]), str(r[3]), 0.0)
                for r in rows[:top_k]
            ]

        doc_tokens: list[list[str]] = [tokenize_bm25(str(r[3])) for r in rows]
        total_docs = len(rows)
        avg_dl = sum(len(toks) for toks in doc_tokens) / max(1, total_docs)

        df: dict[str, int] = {}
        for qt in set(query_terms):
            df[qt] = sum(1 for toks in doc_tokens if qt in toks)

        scored: list[DuckDBChunkRecord] = []
        for row, toks in zip(rows, doc_tokens):
            tf_counter = Counter(toks)
            dl = len(toks)
            bm25_val = 0.0
            for qt in query_terms:
                f = tf_counter.get(qt, 0)
                if f == 0:
                    continue
                n_q = df.get(qt, 0)
                idf = math.log(1.0 + (total_docs - n_q + 0.5) / (n_q + 0.5))
                denom = f + k1 * (1.0 - b + b * (dl / max(1.0, avg_dl)))
                bm25_val += idf * ((f * (k1 + 1.0)) / denom)

            scored.append(
                DuckDBChunkRecord(
                    chunk_id=int(row[0]),
                    doc_source=str(row[1]),
                    section_title=str(row[2]),
                    text=str(row[3]),
                    score=round(bm25_val, 5),
                )
            )

        scored.sort(key=lambda item: (item.score, -item.chunk_id), reverse=True)
        return scored[:top_k]

    def run_custom_sql(self, sql_query: str) -> list[dict[str, Any]]:
        """Execute a SQL query against `rag_chunks` via DuckDB Python or DuckDB CLI."""
        if self.conn is not None:
            rel = self.conn.execute(sql_query)
            cols = [desc[0] for desc in rel.description]
            return [dict(zip(cols, row)) for row in rel.fetchall()]

        # Fallback to `/opt/homebrew/bin/duckdb -light-mode -json`
        setup_stmts = [
            f"CREATE TABLE rag_chunks (chunk_id INTEGER, doc_source VARCHAR, section_title VARCHAR, chunk_text VARCHAR, char_length INTEGER, embedding FLOAT[{self.embedding_dim}]);"
        ]
        for cid, src, sec, txt, clen, vec in self._mem_rows:
            esc_src = src.replace("'", "''")
            esc_sec = sec.replace("'", "''")
            esc_txt = txt.replace("'", "''")
            vec_lit = "[" + ", ".join(f"{v:.5f}" for v in vec) + "]"
            setup_stmts.append(
                f"INSERT INTO rag_chunks VALUES ({cid}, '{esc_src}', '{esc_sec}', '{esc_txt}', {clen}, {vec_lit}::FLOAT[{self.embedding_dim}]);"
            )
        full_sql = "\n".join(setup_stmts) + "\n" + sql_query
        proc = subprocess.run(
            [self._duckdb_cli, "-light-mode", "-json", ":memory:", full_sql],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "DuckDB CLI error")
        out = proc.stdout.strip()
        return json.loads(out) if out else []
