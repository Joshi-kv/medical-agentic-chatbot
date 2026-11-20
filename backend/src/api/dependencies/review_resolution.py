from typing import Annotated

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph

from api.dependencies.appointments import get_appointment_graph
from api.dependencies.human_review import (
    get_human_review_service,
)
from services.appointment_workflow import AppointmentWorkflow
from services.review_resolution import ReviewResolutionService


def get_review_resolution_service(
    appointment_graph: Annotated[
        CompiledStateGraph,
        Depends(get_appointment_graph),
    ],
) -> ReviewResolutionService:
    return ReviewResolutionService(
        human_reviews=get_human_review_service(),
        appointment_workflow=AppointmentWorkflow(
            graph=appointment_graph,
        ),
    )
