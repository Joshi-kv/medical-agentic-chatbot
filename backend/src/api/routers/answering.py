from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies.answering import get_answering_service
from api.dependencies.security import require_scope
from api.schemas.answers import (
    AnswerRequest,
    AnswerResponse,
    CitationResponse,
)
from api.security.api_key import APIPrincipal
from services.answering import AnsweringError, AnsweringService
from services.retrieval import RetrievalServiceError

router = APIRouter(prefix="/v1/answers", tags=["answers"])


@router.post("", response_model=AnswerResponse)
async def create_answer(
    request: AnswerRequest,
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("documents:read")),
    ],
    service: Annotated[
        AnsweringService,
        Depends(get_answering_service),
    ],
) -> AnswerResponse:
    try:
        result = await service.answer(
            query=request.query,
            tenant_id=principal.tenant_id,
            document_id=request.document_id,
        )
    except (AnsweringError, RetrievalServiceError) as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "answering_unavailable",
                "message": "Answering is temporarily unavailable.",
            },
        ) from error

    citations = [
        CitationResponse(
            citation_id=item.citation_id,
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            page_number=item.page_number,
            source=item.source,
            title=item.title,
        )
        for item in result.citations
    ]

    return AnswerResponse(
        answer=result.answer,
        citations=citations,
        sufficient_evidence=result.sufficient_evidence,
        safety_notice=result.safety_notice,
    )
