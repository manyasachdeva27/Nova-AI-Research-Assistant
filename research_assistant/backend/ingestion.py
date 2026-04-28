import os
import logging
from typing import List, Optional
from functools import lru_cache

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import fitz  # pymupdf


FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "volumes/faiss_index")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

logger = logging.getLogger("research_assistant.ingestion")

# Cache the embedding model globally so it's only loaded ONCE
_embeddings_instance = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL}' (one-time)...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
        logger.info("Embedding model loaded and cached.")
    return _embeddings_instance


def load_pdf(file_bytes: bytes, filename: str) -> List[Document]:
    import tempfile
    documents: List[Document] = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": "pdf",
                            "title": filename,
                            "page": page_num + 1,
                        },
                    )
                )
        doc.close()
    finally:
        os.unlink(tmp_path)

    return documents


def load_url(url: str) -> List[Document]:
    loader = WebBaseLoader(url)
    raw_docs = loader.load()
    documents: List[Document] = []
    for doc in raw_docs:
        doc.metadata["source"] = "web"
        doc.metadata["url"] = url
        documents.append(doc)
    return documents


def load_arxiv(query: str, max_results: int = 5) -> List[Document]:
    loader = ArxivLoader(query=query, load_max_docs=max_results)
    raw_docs = loader.load()
    documents: List[Document] = []
    for doc in raw_docs:
        doc.metadata["source"] = "arxiv"
        doc.metadata["authors"] = doc.metadata.get("Authors", "")
        doc.metadata["published"] = doc.metadata.get("Published", "")
        doc.metadata["title"] = doc.metadata.get("Title", "")
        doc.metadata["url"] = doc.metadata.get("Entry ID", "")
        documents.append(doc)
    return documents


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def embed_and_store(chunks: List[Document]) -> FAISS:
    embeddings = _get_embeddings()
    existing_index = load_faiss_index()

    if existing_index is not None:
        existing_index.add_documents(chunks)
        existing_index.save_local(FAISS_INDEX_PATH)
        return existing_index
    else:
        new_index = FAISS.from_documents(chunks, embeddings)
        os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
        new_index.save_local(FAISS_INDEX_PATH)
        return new_index


def load_faiss_index() -> Optional[FAISS]:
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    index_file = os.path.join(FAISS_INDEX_PATH, "index.faiss")
    if not os.path.exists(index_file):
        return None
    try:
        embeddings = _get_embeddings()
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception:
        return None
