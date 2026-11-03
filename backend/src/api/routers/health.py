from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str = "OK"
    service: str = "Medical Chatbot"


@router.get("/", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse
