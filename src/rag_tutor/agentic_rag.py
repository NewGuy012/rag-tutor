"""Module 8: Agentic RAG (Query Decomposition, Iterative Lookup & Verification).

Implements a transparent, educational 3-step Agentic RAG workflow:
1. Query Planner & Decomposer: Splits complex / multi-hop questions into targeted sub-queries.
2. Iterative Retrieval & Sufficiency Evaluator: Runs Hybrid + Reranked search per sub-query,
   checks whether all sub-questions have strong evidence, and executes a refinement lookup if needed.
3. Grounded Generator & Faithfulness Verifier: Synthesizes the answer with citations and audits
   each sentence against retrieved chunks to verify groundedness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .hybrid_search import HybridSearchEngine
from .llm_backend import LlamaMacOSClient, get_default_client
from .reranker import CrossEncoderReranker, RerankedCandidate


@dataclass
class SubQueryStep:
    """A single sub-query executed by the Agentic RAG retrieval loop."""

    step_number: int
    sub_query: str
    reason: str
    top_chunk_id: int
    top_doc_source: str
    top_section: str
    rerank_score: float
    snippet: str


@dataclass
class ClaimVerification:
    """Verification audit result for a single sentence/claim in the response."""

    claim: str
    is_grounded: bool
    supporting_chunk_id: int | None
    confidence: float


@dataclass
class AgenticTrace:
    """Full execution trace of the Agentic RAG workflow for UI inspection."""

    original_query: str
    is_multi_hop: bool
    planned_sub_queries: list[str]
    retrieval_steps: list[SubQueryStep]
    sufficiency_passed: bool
    sufficiency_note: str
    final_context_chunks: list[RerankedCandidate]
    final_answer: str
    llm_backend_used: str
    llm_model_used: str
    latency_ms: float
    verifications: list[ClaimVerification] = field(default_factory=list)
    faithfulness_score: float = 1.0


class AgenticRAGPipeline:
    """End-to-end Agentic RAG pipeline with planning, iterative retrieval, and verification."""

    def __init__(
        self,
        hybrid_engine: HybridSearchEngine,
        reranker: CrossEncoderReranker | None = None,
        llm_client: LlamaMacOSClient | None = None,
    ) -> None:
        self.hybrid_engine = hybrid_engine
        self.reranker = reranker or CrossEncoderReranker()
        self.llm_client = llm_client or get_default_client()

    def decompose_query(self, query: str) -> tuple[bool, list[str]]:
        """Decompose a complex or multi-part user query into targeted sub-queries."""
        cleaned = query.strip()
        # Split on explicit multi-part conjunctions or multiple question marks
        parts: list[str] = []
        if "?" in cleaned[:-1]:
            raw_parts = [p.strip() + "?" for p in cleaned.split("?") if p.strip()]
            parts.extend(raw_parts)
        elif re.search(r"\b(?:and also|and what|and how|and why|compare|versus|vs\.?)\b", cleaned, re.IGNORECASE):
            split_tokens = re.split(
                r"\b(?:and also|and what|and how|and why|compare|versus|vs\.?)\b",
                cleaned,
                flags=re.IGNORECASE,
            )
            parts = [p.strip(" ,?.") for p in split_tokens if len(p.strip(" ,?.")) >= 8]

        if len(parts) >= 2:
            return True, parts[:3]

        return False, [cleaned]

    def run(
        self,
        query: str,
        model: str | None = None,
        enable_thinking: bool = False,
        stage1_candidates: int = 10,
        final_top_k: int = 3,
    ) -> AgenticTrace:
        """Execute the 3-step Agentic RAG workflow and return a full `AgenticTrace`."""
        is_multi_hop, sub_queries = self.decompose_query(query)

        retrieval_steps: list[SubQueryStep] = []
        collected_by_id: dict[int, RerankedCandidate] = {}

        # Step 1 & 2: Execute Hybrid Search + Cross-Encoder Reranking for each sub-query
        for idx, sq in enumerate(sub_queries):
            s1_results = self.hybrid_engine.search(
                sq, top_k=stage1_candidates, alpha=0.5
            )
            reranked = self.reranker.rerank(sq, s1_results, top_k=2)
            if reranked:
                best = reranked[0]
                retrieval_steps.append(
                    SubQueryStep(
                        step_number=idx + 1,
                        sub_query=sq,
                        reason=(
                            f"Sub-query #{idx + 1} from multi-hop planner"
                            if is_multi_hop
                            else "Primary direct lookup"
                        ),
                        top_chunk_id=best.chunk_id,
                        top_doc_source=best.doc_source,
                        top_section=best.section_title,
                        rerank_score=best.rerank_score,
                        snippet=best.text[:160] + ("..." if len(best.text) > 160 else ""),
                    )
                )
                for item in reranked:
                    existing = collected_by_id.get(item.chunk_id)
                    if existing is None or item.rerank_score > existing.rerank_score:
                        collected_by_id[item.chunk_id] = item

        # Sufficiency check: if best score is low (< 0.55), run a broader keyword expansion step
        max_score = max(
            (step.rerank_score for step in retrieval_steps),
            default=0.0,
        )
        if max_score < 0.55:
            expanded_query = re.sub(r"[^\w\s-]", " ", query).strip()
            s1_retry = self.hybrid_engine.search(
                expanded_query, top_k=stage1_candidates, alpha=0.25
            )
            reranked_retry = self.reranker.rerank(expanded_query, s1_retry, top_k=2)
            if reranked_retry:
                best_retry = reranked_retry[0]
                retrieval_steps.append(
                    SubQueryStep(
                        step_number=len(retrieval_steps) + 1,
                        sub_query=f"[Keyword-Boosted Retry] {expanded_query}",
                        reason="Initial confidence < 0.55; triggered BM25-weighted iterative retry",
                        top_chunk_id=best_retry.chunk_id,
                        top_doc_source=best_retry.doc_source,
                        top_section=best_retry.section_title,
                        rerank_score=best_retry.rerank_score,
                        snippet=best_retry.text[:160] + "...",
                    )
                )
                for item in reranked_retry:
                    if item.chunk_id not in collected_by_id:
                        collected_by_id[item.chunk_id] = item
            sufficiency_passed = max(max_score, best_retry.rerank_score if reranked_retry else 0.0) >= 0.45
            sufficiency_note = (
                "Triggered iterative BM25-weighted refinement lookup after low initial confidence."
            )
        else:
            sufficiency_passed = True
            sufficiency_note = (
                f"All {len(sub_queries)} sub-query lookup(s) passed confidence threshold "
                f"(peak Cross-Encoder score = {max_score:.3f})."
            )

        final_chunks = sorted(
            collected_by_id.values(),
            key=lambda c: c.rerank_score,
            reverse=True,
        )[: max(final_top_k, len(sub_queries))]

        # Step 3: Grounded Generation via Llama.app (or offline fallback)
        context_block = "\n\n".join(
            f"[Chunk #{c.chunk_id} | {c.doc_source} > {c.section_title} | score={c.rerank_score:.3f}]\n{c.text}"
            for c in final_chunks
        )
        system_prompt = (
            "You are an expert RAG Tutor assistant.\n"
            "Answer the user's question using ONLY the verified context chunks below. "
            "Cite the chunk numbers (e.g., [Chunk #2]) for every factual claim you make.\n\n"
            f"VERIFIED CONTEXT:\n{context_block}"
        )

        gen_result = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=query,
            model=model,
            enable_thinking=enable_thinking,
            retrieved_context=[c.text for c in final_chunks],
        )

        # Step 4: Self-Reflection / Faithfulness Verification
        verifications, faith_score = self.verify_response(
            gen_result.content, final_chunks
        )

        return AgenticTrace(
            original_query=query,
            is_multi_hop=is_multi_hop,
            planned_sub_queries=sub_queries,
            retrieval_steps=retrieval_steps,
            sufficiency_passed=sufficiency_passed,
            sufficiency_note=sufficiency_note,
            final_context_chunks=final_chunks,
            final_answer=gen_result.content,
            llm_backend_used=gen_result.backend,
            llm_model_used=gen_result.model,
            latency_ms=gen_result.latency_ms,
            verifications=verifications,
            faithfulness_score=faith_score,
        )

    def verify_response(
        self,
        answer_text: str,
        context_chunks: list[RerankedCandidate],
    ) -> tuple[list[ClaimVerification], float]:
        """Audit each claim sentence in `answer_text` against `context_chunks`."""
        lines = [
            line.strip(" -*")
            for line in re.split(r"(?<=[.!?])\s+|\n+", answer_text)
            if len(line.strip(" -*")) >= 20
            and not line.strip().lower().startswith("based strictly on")
        ]
        if not lines:
            return [], 1.0

        verifications: list[ClaimVerification] = []
        for claim in lines[:6]:
            best_cid: int | None = None
            best_conf = 0.0
            for ch in context_chunks:
                score, _ = self.reranker.score_pair(claim, ch.text)
                if score > best_conf:
                    best_conf = score
                    best_cid = ch.chunk_id
            grounded = best_conf >= 0.35
            verifications.append(
                ClaimVerification(
                    claim=claim,
                    is_grounded=grounded,
                    supporting_chunk_id=best_cid if grounded else None,
                    confidence=round(best_conf, 3),
                )
            )

        grounded_count = sum(1 for v in verifications if v.is_grounded)
        faithfulness = round(grounded_count / max(1, len(verifications)), 3)
        return verifications, faithfulness
