from functools import lru_cache

from fastapi import HTTPException, Request, status
from langgraph.graph.state import CompiledStateGraph

from agent.workflows.appointments.nodes import AppointmentNodes
from api.dependencies.human_review import get_human_review_service
from integrations.nvidia_nim import get_nvidia_nim_chat_model
from integrations.supabase import get_supabase_client
from repositories.supabase_appointments import SupabaseAppointmentStore
from services.appointment_extractor import AppointmentExtractor
from services.appointments import AppointmentService


@lru_cache
def get_appointment_store() -> SupabaseAppointmentStore:
    return SupabaseAppointmentStore(
        client=get_supabase_client(),
    )


@lru_cache
def get_appointment_service() -> AppointmentService:
    return AppointmentService(
        store=get_appointment_store(),
    )


@lru_cache
def get_appointment_extractor() -> AppointmentExtractor:
    return AppointmentExtractor(
        chat_model=get_nvidia_nim_chat_model(),
    )


@lru_cache
def get_appointment_nodes() -> AppointmentNodes:
    return AppointmentNodes(
        appointment_service=get_appointment_service(),
        extractor=get_appointment_extractor(),
        human_review_service=get_human_review_service(),
    )


def get_appointment_graph(request: Request) -> CompiledStateGraph:
    graph = getattr(request.app.state, "appointment_graph", None)

    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "appointment_workflow_unavailable",
                "message": "The appointment workflow is unavailable.",
            },
        )

    return graph
