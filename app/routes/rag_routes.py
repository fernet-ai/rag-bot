from fastapi import APIRouter, HTTPException, Header,Request, File, UploadFile
from app.services.rag_service import answer_query, get_relevant_docs, get_info


router = APIRouter()



@router.get("/")
def read_root():
    return {"Hello": "World"}


@router.get("/ai/info")
async def rag_info(request: Request):
    return get_info(request)


@router.get("/ai/ask")
async def get_answer(prompt: str, request: Request):
    return answer_query(request, prompt)


@router.get("/ai/finddocuments")
async def get_docs(prompt: str, request: Request):
    return get_relevant_docs(request, prompt)


@router.post("/uploadDocument/")
async def upload_document(file: UploadFile = File(...)):
   return "file caricato"



