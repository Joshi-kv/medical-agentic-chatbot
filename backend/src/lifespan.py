import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from agent.graph import build_appointment_graph
from api.dependencies.appointments import get_appointment_nodes
from config.settings import get_settings
from integrations.langsmith import configure_langsmith_tracing
from integrations.supabase import build_document_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    checkpoint_pool: AsyncConnectionPool | None = None

    if configure_langsmith_tracing(settings):
        logger.info(
            "LangSmith tracing enabled for project %s",
            settings.langsmith_project,
        )

    document_service = build_document_service(settings)

    if document_service is None:
        logger.warning(
            "Document service disabled: SUPABASE_URL or "
            "SUPABASE_SERVICE_ROLE_KEY is missing"
        )
    else:
        app.state.document_service = document_service

    try:
        if settings.supabase_database_url is None:
            logger.warning(
                "Appointment workflow disabled: SUPABASE_DATABASE_URL is missing"
            )
        else:
            checkpoint_pool = AsyncConnectionPool(
                conninfo=(settings.supabase_database_url.get_secret_value()),
                min_size=1,
                max_size=5,
                open=False,
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                    "row_factory": dict_row,
                    "options": "-c search_path=langgraph,public",
                },
            )

            await checkpoint_pool.open(
                wait=True,
                timeout=30,
            )

            checkpointer = AsyncPostgresSaver(
                checkpoint_pool,
                serde=JsonPlusSerializer(
                    allowed_msgpack_modules=None,
                ),
            )

            await checkpointer.setup()

            app.state.appointment_graph = build_appointment_graph(
                nodes=get_appointment_nodes(),
                checkpointer=checkpointer,
            )

            logger.info("Appointment workflow initialized")

        yield

    finally:
        if checkpoint_pool is not None:
            await checkpoint_pool.close()
