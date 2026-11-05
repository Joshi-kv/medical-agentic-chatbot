from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Response,
    UploadFile,
    status,
)

from api.dependencies.documents import get_document_service
from api.dependencies.security import require_scope
from api.schemas.documents import (
    DeletionAcceptedResponse,
    DocumentAcceptedResponse,
    DocumentResponse,
    DocumentStatus,
)
from api.security.api_key import APIPrincipal
from services.documents import (
    DocumentNotFound,
    DocumentService,
    DocumentServiceError,
    EmptyUpload,
    IdempotencyConflict,
    InvalidDocumentType,
    UploadTooLarge,
)

router = APIRouter(
    prefix="/v1/documents",
    tags=["documents"],
)


@router.post(
    "",
    response_model=DocumentAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF for ingestion",
)
async def upload_document(
    response: Response,
    file: Annotated[UploadFile, File(description="PDF document to ingest")],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=16, max_length=128),
    ],
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("documents:write")),
    ],
    service: Annotated[DocumentService, Depends(get_document_service)],
    title: Annotated[str | None, Form(max_length=200)] = None,
) -> DocumentAcceptedResponse:
    try:
        try:
            document = await service.submit_document(
                content=_read_upload(file),
                filename=file.filename or "document.pdf",
                declared_content_type=file.content_type,
                title=title,
                tenant_id=principal.tenant_id,
                idempotency_key=idempotency_key,
                actor_id=principal.key_id,
            )
        except DocumentServiceError as error:
            _raise_http_error(error)
    finally:
        await file.close()

    status_url = f"/v1/documents/{document.document_id}"
    response.headers["Location"] = status_url
    response.headers["Cache-Control"] = "no-store"

    return DocumentAcceptedResponse(
        document_id=document.document_id,
        status=document.status,
        status_url=status_url,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document ingestion status",
)
async def get_document(
    document_id: str,
    response: Response,
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("documents:read")),
    ],
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DocumentResponse:
    try:
        document = await service.get_document(
            document_id=document_id,
            tenant_id=principal.tenant_id,
        )
    except DocumentServiceError as error:
        _raise_http_error(error)
    response.headers["Cache-Control"] = "no-store"
    return DocumentResponse.model_validate(document)


@router.delete(
    "/{document_id}",
    response_model=DeletionAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request document deletion",
)
async def delete_document(
    document_id: str,
    response: Response,
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("documents:delete")),
    ],
    service: Annotated[DocumentService, Depends(get_document_service)],
) -> DeletionAcceptedResponse:
    try:
        await service.request_deletion(
            document_id=document_id,
            tenant_id=principal.tenant_id,
            actor_id=principal.key_id,
        )
    except DocumentServiceError as error:
        _raise_http_error(error)
    response.headers["Cache-Control"] = "no-store"
    return DeletionAcceptedResponse(
        document_id=document_id,
        status=DocumentStatus.DELETING,
    )


async def _read_upload(
    file: UploadFile,
    chunk_size: int = 1024 * 1024,
) -> AsyncIterator[bytes]:
    while chunk := await file.read(chunk_size):
        yield chunk


def _raise_http_error(error: DocumentServiceError) -> None:
    if isinstance(error, InvalidDocumentType):
        status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        code = "unsupported_media_type"
        message = "Only PDF documents are accepted."
    elif isinstance(error, UploadTooLarge):
        status_code = status.HTTP_413_CONTENT_TOO_LARGE
        code = "upload_too_large"
        message = "The uploaded document exceeds the size limit."
    elif isinstance(error, EmptyUpload):
        status_code = status.HTTP_400_BAD_REQUEST
        code = "empty_upload"
        message = "The uploaded document is empty."
    elif isinstance(error, IdempotencyConflict):
        status_code = status.HTTP_409_CONFLICT
        code = "idempotency_conflict"
        message = "The idempotency key was already used for different content."
    elif isinstance(error, DocumentNotFound):
        status_code = status.HTTP_404_NOT_FOUND
        code = "document_not_found"
        message = "Document not found."
    else:
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        code = "service_unavailable"
        message = "The document service is temporarily unavailable."

    raise HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    ) from error
