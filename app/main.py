# main.py
from fastapi import FastAPI
from app.rag_core import RAGPipeline
from app.bot import ChatUI
from app.routes import bot_routes, rag_routes
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()



# Istanzio una sola vola il modello all'avvio del server
@app.on_event("startup")
async def startup():
    app.state.RAG = RAGPipeline()
    chat_ui = ChatUI()
    chat_ui.start()


app.include_router(rag_routes.router, prefix="/rag", tags=["RAG"])
app.include_router(bot_routes.router, prefix="/bot", tags=["Bot"])
