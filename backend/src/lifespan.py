import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.settings import get_settings
from integrations.supabase import build_document_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    document_service = build_document_service(get_settings())
    if document_service is None:
        logger.warning(
            "Document service disabled: SUPABASE_URL or "
            "SUPABASE_SERVICE_ROLE_KEY is missing"
        )
    else:
        app.state.document_service = document_service

    yield
