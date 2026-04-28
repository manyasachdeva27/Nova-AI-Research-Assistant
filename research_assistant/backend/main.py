import os
import logging
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingestion import load_pdf, load_url, load_arxiv, chunk_documents, embed_and_store
from agent import run_agent
from evaluation import run_pipeline_for_evaluation, evaluate_with_ragas, generate_evaluation_report

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("research_assistant")

app = FastAPI(title="AI Research Assistant", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class URLIngestRequest(BaseModel):
    url: str

class ArxivIngestRequest(BaseModel):
    query: str
    max_results: int = 5

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage] = []

class IngestResponse(BaseModel):
    status: str
    chunks_stored: int

class SourceItem(BaseModel):
    title: str
    url: str
    source_type: str
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    intent: str
    thoughts: List[str] = []
    raw_chunks: List[str] = []


class EvalQuestion(BaseModel):
    question: str
    ground_truth: str


class EvaluateRequest(BaseModel):
    test_data: List[EvalQuestion]
    metrics: Optional[List[str]] = None


class EvaluateResponse(BaseModel):
    aggregate_scores: dict
    per_question_scores: List[dict]
    num_questions: int
    metrics_used: List[str]
    evaluated_at: str
    report: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        filename = file.filename or "uploaded.pdf"
        docs = load_pdf(file_bytes, filename)
        if not docs:
            raise HTTPException(status_code=400, detail="No text extracted from PDF")
        chunks = chunk_documents(docs)
        embed_and_store(chunks)
        return IngestResponse(status="success", chunks_stored=len(chunks))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(request: URLIngestRequest):
    try:
        docs = load_url(request.url)
        if not docs:
            raise HTTPException(status_code=400, detail="No content extracted from URL")
        chunks = chunk_documents(docs)
        embed_and_store(chunks)
        return IngestResponse(status="success", chunks_stored=len(chunks))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/arxiv", response_model=IngestResponse)
async def ingest_arxiv(request: ArxivIngestRequest):
    try:
        docs = load_arxiv(request.query, request.max_results)
        if not docs:
            raise HTTPException(status_code=400, detail="No papers found")
        chunks = chunk_documents(docs)
        embed_and_store(chunks)
        return IngestResponse(status="success", chunks_stored=len(chunks))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ArXiv ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    try:
        chat_history = [{"role": m.role, "content": m.content} for m in request.chat_history]
        result = run_agent(request.question, chat_history)
        sources = [
            SourceItem(title=s.get("title", ""), url=s.get("url", ""),
                       source_type=s.get("source_type", "unknown"),
                       relevance_score=s.get("relevance_score", 0.0))
            for s in result["sources"]
        ]
        return QueryResponse(
            answer=result["answer"], 
            sources=sources, 
            intent=result["intent"], 
            thoughts=result.get("thoughts", []),
            raw_chunks=result.get("raw_chunks", [])
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_rag(request: EvaluateRequest):
    try:
        test_data = [{"question": q.question, "ground_truth": q.ground_truth} for q in request.test_data]
        logger.info(f"Running Ragas evaluation on {len(test_data)} questions")

        pipeline_results = run_pipeline_for_evaluation(test_data)
        eval_results = evaluate_with_ragas(pipeline_results, request.metrics)
        report = generate_evaluation_report(eval_results)

        return EvaluateResponse(
            aggregate_scores=eval_results["aggregate_scores"],
            per_question_scores=eval_results["per_question_scores"],
            num_questions=eval_results["num_questions"],
            metrics_used=eval_results["metrics_used"],
            evaluated_at=eval_results["evaluated_at"],
            report=report,
        )
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
