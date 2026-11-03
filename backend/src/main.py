from fastapi import FastAPI

from api.routers.health import router as health_router

app = FastAPI(
    title="Medical Agentic Chatbot API",
    description="Backend API for the medical agentic chatbot.",
    version="0.1.0",
)

app.include_router(health_router, prefix="/health", tags=["health"])
