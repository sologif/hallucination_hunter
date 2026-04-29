from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from engine import analyze_hallucination
from rag.vector_db import db as vector_db
from rag.generator import generate_answer

app = FastAPI(title="Hallucination Hunter API")

class AnalyzeRequest(BaseModel):
    source_text: str
    generated_text: str

class AskRequest(BaseModel):
    query: str

@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    result = analyze_hallucination(req.source_text, req.generated_text)
    return result

@app.post("/api/ask")
def ask(req: AskRequest):
    # 1. Search Vector DB for context
    sources = vector_db.search(req.query, limit=3)
    
    # 2. Generate answer with strict bounds
    answer = generate_answer(req.query, sources)
    
    if answer.startswith("ERROR:"):
        return {
            "query": req.query,
            "answer": answer,
            "sources": sources,
            "verification": {"verdict": "ERROR", "score": 0.0, "details": "Boundary violation"}
        }

    # 3. Verification Layer
    # Combine retrieved sources into a single source passage for the validation engine
    source_passage = " ".join([s["text"] for s in sources])
    
    # Analyze the generated answer against the retrieved ground truth
    verification_result = analyze_hallucination(source_passage, answer)
    
    return {
        "query": req.query,
        "answer": answer,
        "sources": sources,
        "verification": verification_result
    }

# Serve static files for frontend
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
