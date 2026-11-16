"""DuckDB Embedded Vector & Full-Text Search Store."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import duckdb

from .chunker import Chunk
from .llm_backend import get_default_embedder, tokenize

_READONLY_SQL_RE = re.compile(
    r"^\s*(?:SELECT|WITH|DESCRIBE|SUMMARIZE|EXPLAIN|SHOW)\b",
    re.IGNORECASE,
)
_FORBIDDEN_SQL_RE = re.compile(
    r"\b(?:COPY|ATTACH|DETACH|INSTALL|LOAD|EXPORT|IMPORT|CALL|SET|PRAGMA|"
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|"
    r"read_csv|read_parquet|read_json|read_text|read_blob|glob|st_read)\b",
    re.IGNORECASE,
)


@dataclass
class DuckDBChunkRecord:
    """A scored chunk record returned from DuckDB."""

    chunk_id: int
    doc_source: str
    section_title: str
    text: str
    score: float


class DuckDBVectorStore:
    """Embedded DuckDB vector (`FLOAT[dim]`) and Okapi BM25 store."""

    def __init__(
        self,
        db_path: str = ":memory:",
        embedding_dim: int = 128,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.db_path = db_path
        self.embedding_dim = int(embedding_dim)
        self.embed_fn = embed_fn or get_default_embedder().embed
        self.conn = duckdb.connect(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
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
        if self.db_path == ":memory:":
            self.conn.execute("SET enable_external_access = false;")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> DuckDBVectorStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def clear(self) -> None:
        """Remove all rows from `rag_chunks`."""
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

        if rows:
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
        row = self.conn.execute("SELECT COUNT(*) FROM rag_chunks;").fetchone()
        return int(row[0]) if row else 0

    def search_vector(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[DuckDBChunkRecord]:
        """Run exact cosine similarity search in DuckDB (`array_cosine_similarity`)."""
        top_k = max(1, int(top_k))
        query_vec = self.embed_fn(query)
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
        rows = self.conn.execute(sql, [query_vec, top_k]).fetchall()
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

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> list[DuckDBChunkRecord]:
        """Compute Okapi BM25 (`BM25-K1-1.2`, `b=0.75`) over chunks stored in DuckDB."""
        top_k = max(1, int(top_k))
        rows = self.conn.execute(
            """
            SELECT chunk_id, doc_source, section_title, chunk_text
            FROM rag_chunks
            ORDER BY chunk_id ASC;
            """
        ).fetchall()
        if not rows:
            return []

        query_terms = tokenize(query, stem=False)
        if not query_terms:
            return [
                DuckDBChunkRecord(int(r[0]), str(r[1]), str(r[2]), str(r[3]), 0.0)
                for r in rows[:top_k]
            ]

        doc_tokens = [tokenize(str(r[3]), stem=False) for r in rows]
        total_docs = len(rows)
        avg_dl = sum(len(toks) for toks in doc_tokens) / max(1, total_docs)
        df = {
            qt: sum(1 for toks in doc_tokens if qt in toks)
            for qt in set(query_terms)
        }

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
        """Execute a validated read-only SQL query against `rag_chunks`."""
        stripped = re.sub(r"--[^\n]*|/\*[\s\S]*?\*/", "", sql_query).strip().rstrip(";").strip()
        if not stripped:
            raise ValueError("SQL query cannot be empty.")
        if ";" in stripped:
            raise ValueError("Only a single read-only SQL statement is allowed.")
        if not _READONLY_SQL_RE.match(stripped) or _FORBIDDEN_SQL_RE.search(stripped):
            raise ValueError("Only read-only SELECT/DESCRIBE/SUMMARIZE queries on rag_chunks are permitted.")

        rel = self.conn.execute(stripped)
        cols = [desc[0] for desc in rel.description]
        return [dict(zip(cols, row)) for row in rel.fetchall()]
