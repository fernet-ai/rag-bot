# main.py
from fastapi import FastAPI
from app.rag_core import RAGPipeline
from app.routes import bot_routes, rag_routes

app = FastAPI()


# Istanzio una sola vola il modello all'avvio del server
@app.on_event("startup")
async def startup():
    app.state.RAG = RAGPipeline()


app.include_router(rag_routes.router)
#app.include_router(bot_routes.router)
