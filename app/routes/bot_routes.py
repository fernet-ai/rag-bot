from fastapi import APIRouter, Request, Body
from typing import Dict, Any
from app.services.rag_service import answer_query

router = APIRouter()

@router.post("/ask")
async def gradio_chat(request: Request, data: Dict[str, Any] = Body(...)):
    prompt = data.get("prompt", "")
    if not prompt:
        return {"response": "[Prompt mancante]"}

    response = answer_query(request, prompt)
    return {"response": response}
