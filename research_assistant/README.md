# 🚀 Nova — AI Research Assistant

An intelligent, full-stack **RAG-based AI Research Assistant** that enables users to upload academic papers (PDFs/URLs), ask natural language questions, and receive comprehensive, cited answers with automatic discovery of related research.

## ✨ Features

- **Document Ingestion** — Upload PDFs, scrape web URLs, or search ArXiv directly
- **Semantic Retrieval** — FAISS vector search with HuggingFace `all-MiniLM-L6-v2` embeddings
- **Cohere Reranking** — Cross-encoder reranking for precision-optimized context selection
- **LLM Generation** — Llama-3.3-70b via Groq for detailed, cited answers
- **Related Paper Discovery** — Automatic ArXiv + web search for similar academic papers
- **Evaluation Pipeline** — Ragas framework (faithfulness, answer relevancy, context precision, context recall)
- **Observability** — LangSmith tracing for full pipeline monitoring

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────────────────────────────────┐
│   Streamlit UI  │────▶│              FastAPI Backend                 │
│   (frontend/)   │◀────│                                              │
└─────────────────┘     │  Ingest ──▶ Chunk ──▶ Embed ──▶ FAISS Store │
                        │                                              │
                        │  Query ──▶ Retrieve ──▶ Deduplicate          │
                        │        ──▶ Rerank (Cohere)                   │
                        │        ──▶ ArXiv/Web Search                  │
                        │        ──▶ Generate (Llama-3.3-70b / Groq)   │
                        └──────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit (custom CSS, glassmorphism UI) |
| **Backend** | FastAPI, Uvicorn |
| **LLM** | Llama-3.3-70b via Groq |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Reranking** | Cohere Rerank API |
| **Orchestration** | LangChain |
| **Search** | ArXiv API, Tavily Web Search |
| **Evaluation** | Ragas (faithfulness, relevancy, precision, recall) |
| **Observability** | LangSmith Tracing |

## 📁 Project Structure

```
research_assistant/
├── backend/
│   ├── main.py            # FastAPI app with /ingest and /query endpoints
│   ├── agent.py           # RAG pipeline: retrieve → dedupe → rerank → generate
│   ├── ingestion.py       # PDF/URL/ArXiv document processing & chunking
│   ├── retriever.py       # FAISS vector search
│   ├── reranker.py        # Cohere cross-encoder reranking
│   ├── tools.py           # Web search & ArXiv search tools
│   ├── report.py          # Markdown research report generator
│   ├── evaluation.py      # Ragas evaluation integration
│   └── requirements.txt
├── frontend/
│   ├── app.py             # Streamlit chat UI with thinking process display
│   └── requirements.txt
├── evaluation/
│   ├── evaluate.py        # Standalone Ragas evaluation runner
│   ├── test_dataset.csv   # Test questions with ground truth
│   └── requirements.txt
├── volumes/               # FAISS index storage
├── .env.template          # Required API keys template
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/manyasachdeva27/Nova-AI-Research-Assistant.git
cd Nova-AI-Research-Assistant
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### 2. Configure API Keys
```bash
cp .env.template .env
# Edit .env with your keys:
# GROQ_API_KEY, COHERE_API_KEY, TAVILY_API_KEY, LANGSMITH_API_KEY
```

### 3. Run
```bash
# Terminal 1 — Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
streamlit run app.py
```

## 📊 Evaluation

Run the Ragas evaluation pipeline against the test dataset:

```bash
cd evaluation
python evaluate.py --csv test_dataset.csv --backend http://localhost:8000
```

**Metrics measured:**
- **Faithfulness** — Are answers grounded in retrieved context?
- **Answer Relevancy** — Do answers address the user's question?
- **Context Precision** — Is the retrieved context relevant?
- **Context Recall** — Is all necessary context retrieved?

## 🔑 API Keys Required

| Key | Service | Get it at |
|-----|---------|-----------|
| `GROQ_API_KEY` | LLM inference | [console.groq.com](https://console.groq.com) |
| `COHERE_API_KEY` | Reranking | [cohere.com](https://cohere.com) |
| `TAVILY_API_KEY` | Web search | [tavily.com](https://tavily.com) |
| `LANGSMITH_API_KEY` | Tracing (optional) | [smith.langchain.com](https://smith.langchain.com) |
