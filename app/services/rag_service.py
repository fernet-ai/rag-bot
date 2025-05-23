
from fastapi import Request

def get_info(request: Request) -> dict:
    rag = request.app.state.RAG
    return rag.info()


def answer_query(request: Request, query: str) -> str:
    rag = request.app.state.RAG
    return rag.run(query)


def get_relevant_docs(request: Request, query: str) -> dict:    
    rag = request.app.state.RAG
    return rag.retrieve_relevant_docs(query)