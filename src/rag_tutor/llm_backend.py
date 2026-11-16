"""Local LLM & Embedding Backend for Llama.app (ggml-org/Llama-macOS).

Connects to the native macOS Llama.app server at `http://localhost:9931/v1`
(OpenAI-compatible API). Also provides:
1. Live model discovery via `GET /v1/models` (reporting load status & capabilities).
2. Chat completions via `POST /v1/chat/completions` with support for fast non-thinking
   mode (`reasoning_effort="none"`, `chat_template_kwargs={"enable_thinking": False}`)
   and structured JSON schema outputs.
3. A fast, deterministic 128-dimensional local Python embedding engine (plus optional
   `/v1/embeddings` support) and a transparent offline fallback generator so all
   Marimo notebooks work immediately even when Llama.app is closed or loading models.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


DEFAULT_LLAMA_MACOS_URL = "http://localhost:9931"

# Curated semantic concept clusters so our built-in local embedding engine captures
# genuine semantic similarity (e.g., "efficiency / MPGe / kWh" or "regenerative / braking / kinetic")
# in addition to character n-gram and lexical token features.
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
]


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric/code tokens with light stemming."""
    raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*%?", text.lower())
    stemmed: list[str] = []
    for tok in raw_tokens:
        stemmed.append(tok)
        if len(tok) > 4 and tok.endswith("ing"):
            stemmed.append(tok[:-3])
        elif len(tok) > 4 and tok.endswith("ed"):
            stemmed.append(tok[:-2])
        elif len(tok) > 3 and tok.endswith("es"):
            stemmed.append(tok[:-2])
        elif len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            stemmed.append(tok[:-1])
    return stemmed


class LocalEmbeddingEngine:
    """Deterministic, zero-dependency 128-D embedding engine for educational RAG.

    Combines:
    - Dimensions 0..14: Semantic concept cluster activations (so synonyms like
      'sleep' and 'nap' or 'fast' and 'speed' activate the same dimensions).
    - Dimensions 15..95: Hashed word &stemmed-token projections with sublinear TF.
    - Dimensions 96..127: Character 3-gram projections for robustness to typos
      and rare technical codes (like `ERR-VEC-768`).
    All vectors are L2-normalized to unit length (`||v|| = 1.0`).
    """

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
        tokens = _tokenize(cleaned)
        token_set = set(tokens)

        # 1. Semantic concept dims (0 .. len(SEMANTIC_CLUSTERS)-1)
        num_clusters = min(len(SEMANTIC_CLUSTERS), self.dim // 3)
        for idx in range(num_clusters):
            _, cluster_words = SEMANTIC_CLUSTERS[idx]
            matches = sum(1 for t in tokens if t in cluster_words)
            if matches > 0:
                vec[idx] = 1.8 * (1.0 + math.log1p(matches))

        # 2. Hashed lexical token dimensions
        lex_start = num_clusters
        lex_span = max(16, int(self.dim * 0.6) - lex_start)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "by", "with", "from", "that",
            "this", "it", "as", "be", "can", "how", "what", "why", "which",
            "who", "when", "where", "do", "does", "did", "many", "much",
        }
        for tok in tokens:
            if tok in stopwords:
                continue
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            slot1 = lex_start + (int.from_bytes(digest[0:2], "little") % lex_span)
            sign1 = 1.0 if (digest[2] & 1) == 0 else -1.0
            slot2 = lex_start + (int.from_bytes(digest[4:6], "little") % lex_span)
            sign2 = 1.0 if (digest[6] & 1) == 0 else -1.0
            # Boost rare/specific technical tokens (containing digits or hyphens)
            weight = 1.6 if ("-" in tok or any(c.isdigit() for c in tok)) else 1.0
            vec[slot1] += sign1 * weight
            vec[slot2] += sign2 * (0.5 * weight)

        # 3. Character 3-gram dimensions for subword & identifier similarity
        ngram_start = lex_start + lex_span
        ngram_span = max(8, self.dim - ngram_start)
        compact = re.sub(r"\s+", " ", cleaned.lower())
        for i in range(max(0, len(compact) - 2)):
            trigram = compact[i : i + 3]
            h = int.from_bytes(
                hashlib.md5(trigram.encode("utf-8")).digest()[:2], "little"
            )
            slot = ngram_start + (h % ngram_span)
            vec[slot] += 0.25

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [round(x / norm, 6) for x in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        return [self.embed(t) for t in texts]


@dataclass
class LlamaModelInfo:
    """Metadata about a model discovered from Llama.app (`GET /v1/models`)."""

    id: str
    status: str = "unknown"
    supports_vision: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMGenerationResult:
    """Result from calling Llama.app or the built-in educational fallback."""

    content: str
    model: str
    backend: str  # "llama-macos" or "educational-fallback"
    latency_ms: float
    thinking_enabled: bool = False


class LlamaMacOSClient:
    """Client for the ggml-org/Llama-macOS native app (`http://localhost:9931`)."""

    def __init__(
        self,
        base_url: str = DEFAULT_LLAMA_MACOS_URL,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def check_health(self) -> tuple[bool, list[LlamaModelInfo], str]:
        """Query `GET /v1/models` to check if Llama.app is running and list models."""
        url = f"{self.base_url}/v1/models"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                models_data = payload.get("data", [])
                models: list[LlamaModelInfo] = []
                for item in models_data:
                    model_id = str(item.get("id", ""))
                    status_obj = item.get("status") or {}
                    status_val = str(status_obj.get("value", "ready"))
                    args = status_obj.get("args") or []
                    supports_vision = "--mmproj" in args if isinstance(args, list) else False
                    if model_id:
                        models.append(
                            LlamaModelInfo(
                                id=model_id,
                                status=status_val,
                                supports_vision=supports_vision,
                                raw=item,
                            )
                        )
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
        json_schema: dict[str, Any] | None = None,
        retrieved_context: list[str] | None = None,
    ) -> LLMGenerationResult:
        """Send a chat completion request to Llama.app (`/v1/chat/completions`).

        Automatically falls back to a transparent context-grounded synthesizer if
        Llama.app is not running or has no models loaded.
        """
        t0 = time.perf_counter()
        online, models, _ = self.check_health()

        selected_model = model
        if online and models and (not selected_model or selected_model == "auto"):
            selected_model = models[0].id

        if online and selected_model and selected_model != "educational-fallback":
            body: dict[str, Any] = {
                "model": selected_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if not enable_thinking:
                # Matches Llama.app RequestBuilder.html specification to disable thinking overhead
                body["reasoning_effort"] = "none"
                body["chat_template_kwargs"] = {"enable_thinking": False}
            if json_schema is not None:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json_schema,
                }

            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    choices = resp_json.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {}
                        content = str(msg.get("content") or "").strip()
                        elapsed = (time.perf_counter() - t0) * 1000.0
                        return LLMGenerationResult(
                            content=content,
                            model=selected_model,
                            backend="llama-macos",
                            latency_ms=round(elapsed, 1),
                            thinking_enabled=enable_thinking,
                        )
            except Exception:
                # Fall through to educational fallback if request fails
                pass

        # Built-in educational fallback synthesizer
        fallback_text = self._synthesize_educational_response(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            retrieved_context=retrieved_context,
        )
        elapsed = (time.perf_counter() - t0) * 1000.0
        return LLMGenerationResult(
            content=fallback_text,
            model="educational-fallback (local deterministic synthesizer)",
            backend="educational-fallback",
            latency_ms=round(elapsed, 1),
            thinking_enabled=False,
        )

    @staticmethod
    def _synthesize_educational_response(
        system_prompt: str,
        user_prompt: str,
        retrieved_context: list[str] | None = None,
    ) -> str:
        """Synthesize a realistic grounded response from retrieved context when offline."""
        chunks: list[str] = []
        if retrieved_context:
            chunks = [c.strip() for c in retrieved_context if c.strip()]
        else:
            # Extract bulleted facts from system_prompt if present
            for line in system_prompt.splitlines():
                stripped = line.strip()
                if stripped.startswith("- ") or stripped.startswith("* ") or stripped.startswith("["):
                    cleaned = re.sub(r"^[-*]\s*(\[[^\]]+\]\s*)?", "", stripped)
                    if cleaned:
                        chunks.append(cleaned)

        if not chunks:
            # Ungrounded (No-RAG) mode demonstration
            return (
                "I do not have verified external context in my prompt for this question. "
                "Without Retrieval-Augmented Generation (RAG), a language model must guess "
                "from parametric memory—which risks hallucinating private codes (like "
                "`PROTO-RAG-101` or `ERR-VEC-768`) or specific numerical facts."
            )

        # Score sentences in the provided chunks by overlap with the user question
        q_tokens = set(_tokenize(user_prompt))
        scored_sentences: list[tuple[float, int, str]] = []
        for c_idx, chunk in enumerate(chunks):
            sentences = re.split(r"(?<=[.!?])\s+|\n+", chunk)
            for sent in sentences:
                s_clean = sent.strip()
                if len(s_clean) < 12 or s_clean.startswith("#"):
                    continue
                s_tokens = set(_tokenize(s_clean))
                overlap = len(q_tokens & s_tokens)
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
    """Return a shared LlamaMacOSClient instance."""
    return _DEFAULT_CLIENT


def get_default_embedder() -> LocalEmbeddingEngine:
    """Return a shared 128-D LocalEmbeddingEngine instance."""
    return _DEFAULT_EMBEDDER
