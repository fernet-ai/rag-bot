# main.py
from fastapi import FastAPI
from app.routes import bot_routes, rag_routes

app = FastAPI()
app.include_router(rag_routes.router)
#app.include_router(bot_routes.router)
