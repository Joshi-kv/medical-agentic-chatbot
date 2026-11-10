from dataclasses import dataclass
from uuid import UUID

from services.retrieval import RetrievalMatch, RetrievalService


@dataclass(frozen=True)
class Evidence:
    citation_id: str
    chunk_id: str
    text: str
    document_id: str
    page_number: int
    source: str
    score: float
    title: str | None = None


@dataclass(frozen=True)
class EvidencePackage:
    query: str
    items: list[Evidence]
    sufficient: bool


class EvidenceService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        min_score: float = 0.35,
        candidate_count: int = 15,
        evidence_limit: int = 3,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._min_score = min_score
        self._candidate_count = candidate_count
        self._evidence_limit = evidence_limit

    async def collect_evidence(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
    ) -> EvidencePackage:
        retrieval_result = await self._retrieval_service.retrieve(
            query=query,
            tenant_id=tenant_id,
            top_k=self._candidate_count,
            document_id=str(document_id) if document_id is not None else None,
        )

        unique_matches = self._remove_duplicates(retrieval_result.matches)
        strong_matches = [
            match for match in unique_matches if match.score >= self._min_score
        ]
        selected_matches = strong_matches[: self._evidence_limit]

        evidence = [
            self._create_evidence(match, position)
            for position, match in enumerate(selected_matches, start=1)
        ]

        return EvidencePackage(
            query=retrieval_result.query,
            items=evidence,
            sufficient=bool(evidence),
        )

    def _remove_duplicates(self, matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
        best_matches: dict[str, RetrievalMatch] = {}

        for match in matches:
            duplicate_keys = match.content_hash or match.chunk_id
            existing_match = best_matches.get(duplicate_keys)

            if existing_match is None or match.score > existing_match.score:
                best_matches[duplicate_keys] = match

        return sorted(
            best_matches.values(), key=lambda match: match.score, reverse=True
        )

    def _create_evidence(self, match: RetrievalMatch, position: int) -> Evidence:
        return Evidence(
            citation_id=f"C{position}",
            chunk_id=match.chunk_id,
            text=match.text,
            document_id=match.document_id,
            page_number=match.page_number,
            source=match.source,
            score=match.score,
            title=match.title,
        )
