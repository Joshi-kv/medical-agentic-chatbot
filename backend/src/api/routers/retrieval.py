from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies.retrieval import get_retrieval_service
from api.dependencies.security import require_scope
from api.schemas.retrieval import (
    RetrievalMatchResponse,
    RetrievalRequest,
    RetrievalResponse,
)
from api.security.api_key import APIPrincipal
from services.retrieval import (
    InvalidRetrievalQuery,
    RetrievalService,
    RetrievalServiceError,
)

router = APIRouter(prefix="/v1/retrieval", tags=["retrieval"])


@router.post("/search", response_model=RetrievalResponse)
async def search_documents(
    request: RetrievalRequest,
    principal: Annotated[APIPrincipal, Depends(require_scope("documents:read"))],
    service: Annotated[RetrievalService, Depends(get_retrieval_service)],
) -> RetrievalResponse:
    """Search over retrieval documents"""
    try:
        document_id = (
            str(request.document_id)
            if request.document_id is not None
            else None
        )
        result = await service.retrieve(
            query=request.query,
            tenant_id=principal.tenant_id,
            top_k=request.top_k,
            document_id=document_id,
        )
    except InvalidRetrievalQuery as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_retrieval_query",
                "message": str(error),
            },
        ) from error
    except RetrievalServiceError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "retrieval_unavailable",
                "message": "Retrieval is temporarily unavailable.",
            },
        ) from error

    matches = [
        RetrievalMatchResponse(
            chunk_id=match.chunk_id,
            score=match.score,
            text=match.text,
            document_id=match.document_id,
            page_number=match.page_number,
            chunk_number=match.chunk_number,
            source=match.source,
            title=match.title,
        )
        for match in result.matches
    ]

    return RetrievalResponse(
        query=result.query,
        matches=matches,
    )
