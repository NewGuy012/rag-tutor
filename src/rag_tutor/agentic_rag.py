"""Agentic RAG: Query Decomposition, Iterative Lookup & Faithfulness Verification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .hybrid_search import HybridSearchEngine
from .llm_backend import LlamaMacOSClient, get_default_client, split_sentences
from .reranker import CrossEncoderReranker, RerankedCandidate
from .scratch_rag import build_rag_prompt


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
    llm_model_used: str
    latency_ms: float
    verifications: list[ClaimVerification] = field(default_factory=list)
    faithfulness_score: float = 1.0


_MULTI_HOP_PATTERN = re.compile(
    r"\b(?:and also|and what|and how|and why|compare|versus|vs\.?)\b",
    re.IGNORECASE,
)


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
        if not cleaned:
            return False, []
        parts: list[str] = []
        if "?" in cleaned[:-1]:
            parts = [p.strip() + "?" for p in cleaned.split("?") if p.strip()]
        elif _MULTI_HOP_PATTERN.search(cleaned):
            parts = [
                p.strip(" ,?.")
                for p in _MULTI_HOP_PATTERN.split(cleaned)
                if len(p.strip(" ,?.")) >= 8
            ]
        return (True, parts[:3]) if len(parts) >= 2 else (False, [cleaned])

    def _retrieve_step(
        self,
        sub_query: str,
        reason: str,
        alpha: float,
        stage1_candidates: int,
        step_number: int,
        collected_by_id: dict[int, RerankedCandidate],
        display_query: str | None = None,
    ) -> SubQueryStep | None:
        s1_results = self.hybrid_engine.search(sub_query, top_k=stage1_candidates, alpha=alpha)
        reranked = self.reranker.rerank(sub_query, s1_results, top_k=2)
        if not reranked:
            return None
        for item in reranked:
            existing = collected_by_id.get(item.chunk_id)
            if existing is None or item.rerank_score > existing.rerank_score:
                collected_by_id[item.chunk_id] = item
        best = reranked[0]
        return SubQueryStep(
            step_number=step_number,
            sub_query=display_query or sub_query,
            reason=reason,
            top_chunk_id=best.chunk_id,
            top_doc_source=best.doc_source,
            top_section=best.section_title,
            rerank_score=best.rerank_score,
        )

    def run(
        self,
        query: str,
        model: str | None = None,
        enable_thinking: bool = False,
        stage1_candidates: int = 10,
        final_top_k: int = 3,
    ) -> AgenticTrace:
        """Execute the Agentic RAG workflow and return a full `AgenticTrace`."""
        cleaned_query = query.strip()
        if not cleaned_query:
            return AgenticTrace(
                original_query="",
                is_multi_hop=False,
                planned_sub_queries=[],
                retrieval_steps=[],
                sufficiency_passed=False,
                sufficiency_note="Empty query skipped.",
                final_context_chunks=[],
                final_answer="Please enter a question to query the RAG pipeline.",
                llm_model_used="educational-fallback",
                latency_ms=0.0,
            )

        is_multi_hop, sub_queries = self.decompose_query(cleaned_query)
        retrieval_steps: list[SubQueryStep] = []
        collected_by_id: dict[int, RerankedCandidate] = {}

        for idx, sq in enumerate(sub_queries):
            reason = f"Sub-query #{idx + 1} from multi-hop planner" if is_multi_hop else "Primary direct lookup"
            step = self._retrieve_step(
                sq, reason, 0.5, stage1_candidates, len(retrieval_steps) + 1, collected_by_id
            )
            if step is not None:
                retrieval_steps.append(step)

        max_score = max((s.rerank_score for s in retrieval_steps), default=0.0)
        if max_score < 0.55:
            expanded_query = re.sub(r"[^\w\s-]", " ", cleaned_query).strip()
            retry_step = self._retrieve_step(
                expanded_query,
                "Initial confidence < 0.55; triggered BM25-weighted iterative retry",
                0.25,
                stage1_candidates,
                len(retrieval_steps) + 1,
                collected_by_id,
                display_query=f"[Keyword-Boosted Retry] {expanded_query}",
            )
            if retry_step is not None:
                retrieval_steps.append(retry_step)
                max_score = max(max_score, retry_step.rerank_score)
            sufficiency_passed = max_score >= 0.45
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
        )[: max(1, final_top_k, len(sub_queries))]

        context_blocks = [
            f"[Chunk #{c.chunk_id} | {c.doc_source} > {c.section_title} | score={c.rerank_score:.3f}]\n{c.text}"
            for c in final_chunks
        ]
        system_prompt, user_prompt = build_rag_prompt(cleaned_query, context_blocks, cite_chunks=True)
        gen_result = self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            enable_thinking=enable_thinking,
            retrieved_context=[c.text for c in final_chunks],
        )
        verifications, faith_score = self.verify_response(gen_result.content, final_chunks)

        return AgenticTrace(
            original_query=cleaned_query,
            is_multi_hop=is_multi_hop,
            planned_sub_queries=sub_queries,
            retrieval_steps=retrieval_steps,
            sufficiency_passed=sufficiency_passed,
            sufficiency_note=sufficiency_note,
            final_context_chunks=final_chunks,
            final_answer=gen_result.content,
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
            cleaned
            for line in split_sentences(answer_text, min_len=20)
            if len(cleaned := line.strip(" -*")) >= 20
            and not cleaned.lower().startswith("based strictly on")
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
        return verifications, round(grounded_count / len(verifications), 3)
