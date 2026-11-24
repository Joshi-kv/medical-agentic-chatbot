from typing import TypedDict

from services.answering import AnswerResult
from services.evidence import Evidence


class ContextMessageState(TypedDict):
    role: str
    content: str


class CitationState(TypedDict):
    citation_id: str
    chunk_id: str
    text: str
    document_id: str
    page_number: int
    source: str
    score: float
    title: str | None


class MedicalAgentState(TypedDict, total=False):
    tenant_id: str
    conversation_id: str | None
    run_id: str | None

    query: str
    document_id: str | None
    conversation_history: list[ContextMessageState]
    streaming: bool

    standalone_query: str
    active_appointment: bool
    intent: str
    follow_up_question: str | None

    answer: str
    citations: list[CitationState]
    sufficient_evidence: bool
    safety_notice: str
    answer_route: str
    review_case_id: str | None


def answer_result_to_state(result: AnswerResult) -> dict[str, object]:
    citations: list[CitationState] = [
        {
            "citation_id": citation.citation_id,
            "chunk_id": citation.chunk_id,
            "text": citation.text,
            "document_id": citation.document_id,
            "page_number": citation.page_number,
            "source": citation.source,
            "score": citation.score,
            "title": citation.title,
        }
        for citation in result.citations
    ]

    return {
        "answer": result.answer,
        "citations": citations,
        "sufficient_evidence": result.sufficient_evidence,
        "safety_notice": result.safety_notice,
        "answer_route": result.route,
        "review_case_id": result.review_case_id,
    }


def answer_result_from_state(state: MedicalAgentState) -> AnswerResult:
    answer = state.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Medical-agent graph completed without an answer.")

    citations = [
        Evidence(
            citation_id=item["citation_id"],
            chunk_id=item["chunk_id"],
            text=item["text"],
            document_id=item["document_id"],
            page_number=item["page_number"],
            source=item["source"],
            score=item["score"],
            title=item["title"],
        )
        for item in state.get("citations", [])
    ]

    return AnswerResult(
        answer=answer,
        citations=citations,
        sufficient_evidence=state.get("sufficient_evidence", False),
        safety_notice=state.get("safety_notice", ""),
        route=state.get("answer_route", "unknown"),
        review_case_id=state.get("review_case_id"),
    )
