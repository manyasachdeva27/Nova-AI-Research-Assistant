import os
from typing import List

import cohere


def rerank(query: str, documents: List[dict], top_n: int = 5) -> List[dict]:
    if not documents:
        return []

    api_key = os.getenv("COHERE_API_KEY", "")
    client = cohere.ClientV2(api_key=api_key)

    doc_texts: List[str] = []
    for doc in documents:
        text = doc.get("content", "")
        if not text:
            text = doc.get("title", "No content")
        doc_texts.append(text)

    top_n = min(top_n, len(documents))

    response = client.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=doc_texts,
        top_n=top_n,
    )

    reranked: List[dict] = []
    for result in response.results:
        idx = result.index
        original_doc = documents[idx].copy()
        original_doc["relevance_score"] = result.relevance_score
        reranked.append(original_doc)

    return reranked
