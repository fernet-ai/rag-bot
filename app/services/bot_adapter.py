from fastapi import Request


def answer_query(request: Request, query: str, history: list = None) -> str:
    rag = request.app.state.RAG
    return rag.run(query, history=history)