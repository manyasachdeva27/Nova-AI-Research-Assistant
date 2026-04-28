from typing import List
from ingestion import load_faiss_index


def search_faiss(query: str, k: int = 10) -> List[dict]:
    index = load_faiss_index()
    if index is None:
        return []

    results = index.similarity_search_with_score(query, k=k)
    documents: List[dict] = []
    for doc, score in results:
        documents.append(
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "metadata": doc.metadata,
                "score": float(score),
            }
        )
    return documents
