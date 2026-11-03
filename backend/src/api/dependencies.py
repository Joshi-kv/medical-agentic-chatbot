from collections.abc import Callable

from fastapi import HTTPException, Request, status

from api.security.api_key import APIPrincipal, AuthenticatedPrincipal
from services.documents import DocumentService


def get_document_service(request: Request) -> DocumentService:
    service = getattr(request.app.state, "document_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "service_unavailable",
                "message": "Document service is unavailable.",
            },
        )
    return service


def require_scope(
    required_scope: str,
) -> Callable[[AuthenticatedPrincipal], APIPrincipal]:
    """Create a dependency that requires one API-key scope."""

    def authorize(
        principal: AuthenticatedPrincipal,
    ) -> APIPrincipal:
        if required_scope not in principal.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "insufficient_scope",
                    "message": "The credential cannot perform this action.",
                },
            )

        return principal

    return authorize
