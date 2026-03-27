from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.rag_service import ask_rag

router = APIRouter()

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(5, ge=1, le=10)

@router.post("/ask")
def ask(req: AskRequest):
    return ask_rag(req.question, top_k=req.top_k)