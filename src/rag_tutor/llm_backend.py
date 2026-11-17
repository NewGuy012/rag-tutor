"""Local LLM & Embedding Backend for Llama.app (ggml-org/Llama-macOS)."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .scratch_rag import vector_norm

DEFAULT_LLAMA_MACOS_URL = "http://localhost:9931"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_MAX_RESPONSE_BYTES = 1_048_576


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirectHandler)

STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
    "to", "for", "of", "and", "or", "by", "with", "from", "that",
    "this", "it", "as", "be", "can", "how", "what", "why", "which",
    "who", "when", "where", "do", "does", "did", "many", "much",
    "should", "i", "my", "our", "use", "used",
})

SEMANTIC_CLUSTERS: list[tuple[str, set[str]]] = [
    (
        "ev_efficiency_mpge",
        {
            "efficiency", "efficient", "mpge", "gallon", "gasoline", "combustion",
            "energy", "85%", "90%", "33.7", "135", "kwh", "miles", "consumption",
            "aerodynamic", "drag", "highway", "velocity", "emissions", "carbon",
        },
    ),
    (
        "ev_regen_braking_driving",
        {
            "regenerative", "braking", "brake", "brakes", "deceleration", "kinetic",
            "generator", "70%", "friction", "pedal", "one-pedal", "rotors", "pads",
            "100000", "150000", "slowing", "stop",
        },
    ),
    (
        "ev_battery_chemistry",
        {
            "battery", "batteries", "chemistry", "lfp", "phosphate", "iron",
            "nmc", "nca", "nickel", "cobalt", "manganese", "solid-state", "solid",
            "electrolyte", "anode", "density", "wh", "kg", "degradation", "cycles",
        },
    ),
    (
        "ev_charging_800v_nacs",
        {
            "charging", "charge", "charger", "fast", "dc", "800-volt", "800v",
            "400-volt", "350", "kw", "nacs", "j3400", "connector", "soc", "80%",
            "taper", "plating", "preconditioning", "bidirectional", "v2h", "v2g",
            "blackout", "household", "grid",
        },
    ),
    (
        "ev_motors_torque_powertrain",
        {
            "motor", "motors", "torque", "rpm", "instantaneous", "transmission",
            "pmsm", "permanent", "magnet", "neodymium", "synchronous", "induction",
            "electromagnetic", "freewheel", "cogging", "wheel", "wheels",
        },
    ),
    (
        "ev_thermal_winter_heatpump",
        {
            "cold", "winter", "freezing", "weather", "temperature", "range",
            "heat", "pump", "pumps", "cabin", "heating", "heater", "ptc",
            "resistive", "viscosity", "ambient", "thermal",
        },
    ),
    (
        "ev_history_space_lunar",
        {
            "history", "1899", "jamais", "contente", "jenatzy", "belgian", "100",
            "105.88", "first", "apollo", "lunar", "roving", "moon", "1971", "1972",
            "silver-zinc",
        },
    ),
    (
        "ev_safety_bms_electrical",
        {
            "bms", "management", "isolation", "chassis", "contactor", "contactors",
            "pyrotechnic", "collision", "orange", "high-voltage", "12-volt",
            "auxiliary", "dc-to-dc", "converter",
        },
    ),
    (
        "rag_hallucination_grounding",
        {
            "rag", "retrieval", "augmented", "generation", "hallucinate",
            "hallucination", "cutoff", "private", "grounding", "grounded",
            "context", "prompt", "parametric", "proto-rag-101", "lewis",
        },
    ),
    (
        "vectors_cosine_embeddings",
        {
            "vector", "vectors", "embedding", "embeddings", "cosine", "similarity",
            "dot", "product", "norm", "magnitude", "angle", "euclidean", "bge",
            "dimension", "768", "384", "128", "err-vec-768", "err-chunk-004",
        },
    ),
    (
        "duckdb_sql_storage",
        {
            "duckdb", "sql", "database", "table", "array_cosine_similarity",
            "float", "vss", "hnsw", "fts", "index", "persistent", "memory",
        },
    ),
    (
        "hybrid_bm25_rrf",
        {
            "hybrid", "bm25", "keyword", "lexical", "semantic", "rrf",
            "reciprocal", "rank", "fusion", "rrf-k60", "bm25-k1-1.2", "k1",
            "exact", "code", "identifier",
        },
    ),
    (
        "reranking_cross_encoder",
        {
            "rerank", "reranker", "reranking", "cross-encoder", "bi-encoder",
            "ms-marco", "stage", "stage2-rerank-15to3", "noise", "distractor",
            "precision", "lost", "middle", "attention",
        },
    ),
    (
        "agentic_rag_workflow",
        {
            "agent", "agentic", "multi-hop", "multi-step", "decompose",
            "decomposition", "planner", "sub-query", "iterative", "verify",
            "verification", "verifier", "faithfulness", "audit", "citation",
        },
    ),
    (
        "chunking_splitting",
        {
            "chunk", "chunks", "chunking", "split", "splitting", "overlap",
            "boundary", "window", "token", "tokens", "section", "markdown",
            "character", "paragraph",
        },
    ),
    (
        "graph_rag_knowledge",
        {
            "graph", "graphrag", "knowledge", "entity", "entities", "triple",
            "triples", "relation", "relationship", "relationships", "hop",
            "hops", "traverse", "traversal", "subgraph", "node", "edge",
        },
    ),
    (
        "modular_query_expansion",
        {
            "modular", "hyde", "hypothetical", "multi-query", "expansion",
            "expand", "convex", "linear", "normalization", "splade", "dense",
            "sparse", "metadata", "filter", "filtering", "qdrant", "weaviate",
            "pinecone", "elasticsearch", "cohere", "convex-alpha-70",
        },
    ),
]


def tokenize(
    text: str,
    *,
    stem: bool = True,
    drop_stopwords: bool = False,
) -> list[str]:
    """Tokenize text into lowercase alphanumeric/hyphenated tokens."""
    pattern = r"[a-z0-9]+(?:-[a-z0-9]+)*%?" if stem else r"[a-z0-9]+(?:-[a-z0-9]+)*"
    raw_tokens = re.findall(pattern, text.lower())
    if not stem:
        if drop_stopwords:
            return [t for t in raw_tokens if t not in STOPWORDS and len(t) > 1]
        return raw_tokens

    tokens: list[str] = []
    for tok in raw_tokens:
        if drop_stopwords and tok in STOPWORDS:
            continue
        tokens.append(tok)
        if len(tok) > 4 and tok.endswith("ing"):
            tokens.append(tok[:-3])
        elif len(tok) > 4 and tok.endswith("ed"):
            tokens.append(tok[:-2])
        elif len(tok) > 3 and tok.endswith("es"):
            tokens.append(tok[:-2])
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tokens.append(tok[:-1])
    return tokens


def split_sentences(text: str, min_len: int = 12) -> list[str]:
    """Split text into sentences/lines with length at least `min_len`."""
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(s.strip()) >= min_len
    ]


class LocalEmbeddingEngine:
    """Deterministic 128-D local embedding engine (concept clusters + token hashes + 3-grams)."""

    def __init__(self, dim: int = 128) -> None:
        if dim < 32:
            raise ValueError("Embedding dimension must be at least 32.")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        """Compute a unit-normalized float vector of length `self.dim`."""
        cleaned = text.strip()
        if not cleaned:
            return [0.0] * self.dim

        vec = [0.0] * self.dim
        tokens = tokenize(cleaned)

        num_clusters = min(len(SEMANTIC_CLUSTERS), self.dim // 3)
        for idx in range(num_clusters):
            _, cluster_words = SEMANTIC_CLUSTERS[idx]
            matches = sum(1 for t in tokens if t in cluster_words)
            if matches > 0:
                vec[idx] = 1.8 * (1.0 + math.log1p(matches))

        lex_start = num_clusters
        lex_span = max(16, int(self.dim * 0.6) - lex_start)
        for tok in tokens:
            if tok in STOPWORDS:
                continue
            digest = hashlib.md5(tok.encode("utf-8"), usedforsecurity=False).digest()
            slot1 = lex_start + (int.from_bytes(digest[0:2], "little") % lex_span)
            sign1 = 1.0 if (digest[2] & 1) == 0 else -1.0
            slot2 = lex_start + (int.from_bytes(digest[4:6], "little") % lex_span)
            sign2 = 1.0 if (digest[6] & 1) == 0 else -1.0
            weight = 1.6 if ("-" in tok or any(c.isdigit() for c in tok)) else 1.0
            vec[slot1] += sign1 * weight
            vec[slot2] += sign2 * (0.5 * weight)

        ngram_start = lex_start + lex_span
        ngram_span = max(8, self.dim - ngram_start)
        compact = re.sub(r"\s+", " ", cleaned.lower())
        for i in range(max(0, len(compact) - 2)):
            trigram = compact[i : i + 3]
            h = int.from_bytes(
                hashlib.md5(trigram.encode("utf-8"), usedforsecurity=False).digest()[:2],
                "little",
            )
            vec[ngram_start + (h % ngram_span)] += 0.25

        norm = vector_norm(vec)
        if norm > 0.0:
            vec = [round(x / norm, 6) for x in vec]
        return vec


@dataclass
class LlamaModelInfo:
    """Model metadata returned by `GET /v1/models`."""

    id: str
    status: str = "unknown"


@dataclass
class LLMGenerationResult:
    """Result from Llama.app or the offline fallback synthesizer."""

    content: str
    model: str
    backend: str
    latency_ms: float
    thinking_enabled: bool = False


class LlamaMacOSClient:
    """Client for the local ggml-org/Llama-macOS server."""

    def __init__(
        self,
        base_url: str = DEFAULT_LLAMA_MACOS_URL,
        timeout_seconds: float = 45.0,
    ) -> None:
        cleaned_url = base_url.rstrip("/")
        parsed = urllib.parse.urlparse(cleaned_url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.netloc
            or parsed.hostname not in _LOOPBACK_HOSTS
            or any((parsed.username, parsed.password, parsed.params, parsed.query, parsed.fragment))
        ):
            raise ValueError(f"Invalid loopback HTTP(S) base_url: {base_url!r}")
        self.base_url = cleaned_url
        self.timeout_seconds = max(0.5, float(timeout_seconds))

    def _request_json(
        self,
        path: str,
        timeout: float,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("Response exceeded 1 MiB safety limit.")
        return json.loads(raw.decode("utf-8"))

    def check_health(self) -> tuple[bool, list[LlamaModelInfo], str]:
        """Query `GET /v1/models` to verify Llama.app availability."""
        try:
            payload = self._request_json("/v1/models", timeout=2.5)
            models = [
                LlamaModelInfo(
                    id=model_id,
                    status=str((item.get("status") or {}).get("value", "ready")),
                )
                for item in payload.get("data", [])
                if (model_id := str(item.get("id", "")))
            ]
            models.sort(key=lambda m: m.id)
            return (
                True,
                models,
                f"Connected to Llama.app at {self.base_url} ({len(models)} model(s) found)",
            )
        except Exception as exc:
            return (
                False,
                [],
                f"Llama.app server not detected at {self.base_url} ({exc.__class__.__name__}). "
                "Using built-in educational fallback mode so all interactive demos remain runnable.",
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        enable_thinking: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 512,
        retrieved_context: list[str] | None = None,
    ) -> LLMGenerationResult:
        """Send a chat completion request to Llama.app with offline fallback."""
        t0 = time.perf_counter()
        selected_model = model

        if selected_model != "educational-fallback":
            if not selected_model or selected_model == "auto":
                online, models, _ = self.check_health()
                selected_model = models[0].id if (online and models) else None

            if selected_model:
                body: dict[str, Any] = {
                    "model": selected_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": max(0.0, float(temperature)),
                    "max_tokens": max(1, int(max_tokens)),
                    "stream": False,
                }
                if not enable_thinking:
                    body["reasoning_effort"] = "none"
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                try:
                    resp_json = self._request_json(
                        "/v1/chat/completions",
                        timeout=self.timeout_seconds,
                        payload=body,
                    )
                    choices = resp_json.get("choices") or []
                    if choices:
                        content = str((choices[0].get("message") or {}).get("content") or "").strip()
                        return LLMGenerationResult(
                            content=content,
                            model=selected_model,
                            backend="llama-macos",
                            latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                            thinking_enabled=enable_thinking,
                        )
                except Exception:
                    pass

        fallback_text = self._synthesize_educational_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retrieved_context=retrieved_context,
        )
        return LLMGenerationResult(
            content=fallback_text,
            model="educational-fallback (local deterministic synthesizer)",
            backend="educational-fallback",
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            thinking_enabled=False,
        )

    @staticmethod
    def _synthesize_educational_response(
        system_prompt: str,
        user_prompt: str,
        retrieved_context: list[str] | None = None,
    ) -> str:
        """Synthesize a grounded response from retrieved context when offline."""
        chunks: list[str] = []
        if retrieved_context:
            chunks = [c.strip() for c in retrieved_context if c.strip()]
        else:
            for line in system_prompt.splitlines():
                stripped = line.strip()
                if stripped.startswith(("- ", "* ", "[")):
                    cleaned = re.sub(r"^[-*]\s*(\[[^\]]+\]\s*)?", "", stripped)
                    if cleaned:
                        chunks.append(cleaned)

        if not chunks:
            return (
                "I do not have verified external context in my prompt for this question. "
                "Without Retrieval-Augmented Generation (RAG), a language model must guess "
                "from parametric memory—which risks hallucinating private codes (like "
                "`PROTO-RAG-101` or `ERR-VEC-768`) or specific numerical facts."
            )

        q_tokens = set(tokenize(user_prompt))
        scored_sentences: list[tuple[float, int, str]] = []
        for c_idx, chunk in enumerate(chunks):
            for s_clean in split_sentences(chunk, min_len=12):
                if s_clean.startswith("#"):
                    continue
                overlap = len(q_tokens & set(tokenize(s_clean)))
                score = overlap + (0.25 if c_idx == 0 else 0.0)
                scored_sentences.append((score, c_idx + 1, s_clean))

        scored_sentences.sort(key=lambda item: item[0], reverse=True)
        top_picks = scored_sentences[:3] if scored_sentences else [(1.0, 1, chunks[0])]

        bullet_lines = [
            f"- {sent} *(Source: Retrieved Context #{src_idx})*"
            for _, src_idx, sent in top_picks
        ]
        return (
            "Based strictly on the retrieved context provided in the prompt:\n\n"
            + "\n".join(bullet_lines)
        )


_DEFAULT_CLIENT = LlamaMacOSClient()
_DEFAULT_EMBEDDER = LocalEmbeddingEngine(dim=128)


def get_default_client() -> LlamaMacOSClient:
    return _DEFAULT_CLIENT


def get_default_embedder() -> LocalEmbeddingEngine:
    return _DEFAULT_EMBEDDER
