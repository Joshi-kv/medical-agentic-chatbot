from collections.abc import Callable

from fastapi import HTTPException, status

from api.security.api_key import APIPrincipal, AuthenticatedPrincipal


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
