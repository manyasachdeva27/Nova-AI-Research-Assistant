import os
from typing import List

from tavily import TavilyClient
from langchain_community.document_loaders import ArxivLoader


def search_web(query: str) -> List[dict]:
    api_key = os.getenv("TAVILY_API_KEY", "")
    client = TavilyClient(api_key=api_key)
    response = client.search(query=query, max_results=5)
    results: List[dict] = []
    for item in response.get("results", []):
        results.append(
            {
                "content": item.get("content", ""),
                "url": item.get("url", ""),
                "title": item.get("title", ""),
            }
        )
    return results


def search_arxiv(query: str, max_results: int = 5) -> List[dict]:
    loader = ArxivLoader(query=query, load_max_docs=max_results)
    docs = loader.load()
    results: List[dict] = []
    for doc in docs:
        results.append(
            {
                "content": doc.page_content[:2000],
                "title": doc.metadata.get("Title", ""),
                "authors": doc.metadata.get("Authors", ""),
                "published": doc.metadata.get("Published", ""),
                "url": doc.metadata.get("Entry ID", ""),
            }
        )
    return results
