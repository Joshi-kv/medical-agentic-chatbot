from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies.human_review import get_human_review_service
from api.dependencies.security import require_scope
from api.schemas.human_review import (
    HumanReviewCaseResponse,
    PatientHumanReviewCaseResponse,
    ResolveHumanReviewRequest,
)
from api.security.api_key import APIPrincipal, StaffPrincipal
from services.human_review import (
    HumanReviewCase,
    HumanReviewConflict,
    HumanReviewError,
    HumanReviewNotFound,
    HumanReviewService,
    InvalidHumanReviewInput,
    ReviewStatus,
)

router = APIRouter(
    prefix="/v1/human-review/cases",
    tags=["human-review"],
)


@router.get(
    "",
    response_model=list[HumanReviewCaseResponse],
)
async def list_review_cases(
    principal: StaffPrincipal,
    service: Annotated[
        HumanReviewService,
        Depends(get_human_review_service),
    ],
    case_status: Annotated[ReviewStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[HumanReviewCaseResponse]:
    try:
        cases = await service.list_cases(
            tenant_id=principal.tenant_id,
            status=case_status,
            limit=limit,
        )
    except HumanReviewError as error:
        raise _review_http_error(error) from error

    return [_response(case) for case in cases]


@router.get(
    "/{case_id}",
    response_model=PatientHumanReviewCaseResponse,
)
async def get_review_case(
    case_id: UUID,
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("conversations:read")),
    ],
    service: Annotated[
        HumanReviewService,
        Depends(get_human_review_service),
    ],
) -> PatientHumanReviewCaseResponse:
    try:
        case = await service.get(
            case_id=str(case_id),
            tenant_id=principal.tenant_id,
        )
    except HumanReviewError as error:
        raise _review_http_error(error) from error

    return _patient_response(case)


@router.post(
    "/{case_id}/claim",
    response_model=HumanReviewCaseResponse,
)
async def claim_review_case(
    case_id: UUID,
    principal: StaffPrincipal,
    service: Annotated[
        HumanReviewService,
        Depends(get_human_review_service),
    ],
) -> HumanReviewCaseResponse:
    try:
        case = await service.claim(
            case_id=str(case_id),
            tenant_id=principal.tenant_id,
            staff_id=principal.key_id,
        )
    except HumanReviewError as error:
        raise _review_http_error(error) from error

    return _response(case)


@router.post(
    "/{case_id}/resolve",
    response_model=HumanReviewCaseResponse,
)
async def resolve_review_case(
    case_id: UUID,
    request: ResolveHumanReviewRequest,
    principal: StaffPrincipal,
    service: Annotated[
        HumanReviewService,
        Depends(get_human_review_service),
    ],
) -> HumanReviewCaseResponse:
    try:
        case = await service.resolve(
            case_id=str(case_id),
            tenant_id=principal.tenant_id,
            staff_id=principal.key_id,
            decision=request.decision,
            resolution=request.resolution,
        )
    except HumanReviewError as error:
        raise _review_http_error(error) from error

    return _response(case)


def _response(case: HumanReviewCase) -> HumanReviewCaseResponse:
    return HumanReviewCaseResponse.model_validate(
        case,
        from_attributes=True,
    )


def _patient_response(case: HumanReviewCase) -> PatientHumanReviewCaseResponse:
    message = None
    if case.resolution is not None:
        resolved_message = case.resolution.get("message")
        if isinstance(resolved_message, str):
            message = resolved_message

    return PatientHumanReviewCaseResponse(
        case_id=case.case_id,
        conversation_id=case.conversation_id,
        run_id=case.run_id,
        review_type=case.review_type,
        status=case.status,
        decision=case.decision,
        message=message,
        expires_at=case.expires_at,
        created_at=case.created_at,
        updated_at=case.updated_at,
    )


def _review_http_error(error: HumanReviewError) -> HTTPException:
    if isinstance(error, HumanReviewNotFound):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "review_case_not_found",
                "message": "Human-review case not found.",
            },
        )

    if isinstance(error, HumanReviewConflict):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "review_case_conflict",
                "message": str(error),
            },
        )

    if isinstance(error, InvalidHumanReviewInput):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_review_request",
                "message": str(error),
            },
        )

    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "human_review_unavailable",
            "message": "Human-review service is temporarily unavailable.",
        },
    )
