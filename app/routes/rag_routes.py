from fastapi import APIRouter,File, UploadFile

router = APIRouter()


@router.get("/")
def read_root():
    return {"Hello": "World"}


@router.get("/ue")
def read_root():
    return "ue"


@router.post("/uploadDocument/")
async def upload_document(file: UploadFile = File(...)):
   return "file caricato"