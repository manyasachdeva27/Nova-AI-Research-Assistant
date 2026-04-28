import os
from typing import List

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langsmith import traceable

from retriever import search_faiss
from reranker import rerank
from tools import search_web, search_arxiv


GROQ_MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2


def _get_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _deduplicate_chunks(results: List[dict]) -> List[dict]:
    """Remove near-duplicate chunks (same page, same source)."""
    seen_keys = set()
    unique = []
    for doc in results:
        source = doc.get("title", "") or doc.get("source", "")
        page = doc.get("metadata", {}).get("page", "")
        key = f"{source}::page{page}"
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(doc)
    return unique


def _extract_paper_topics(context: str) -> str:
    """Use a fast LLM call to extract the paper's core topics for searching similar papers."""
    llm = _get_llm()
    msg = llm.invoke([
        SystemMessage(content="You extract search keywords from academic text. Return ONLY a short search query (max 15 words) that captures the paper's core topic, methods, and domain. No explanation."),
        HumanMessage(content=f"Extract a search query for finding similar academic papers based on this text:\n\n{context[:3000]}")
    ])
    return msg.content.strip().strip('"').strip("'")


SYSTEM_PROMPT = """You are Nova, an expert AI research assistant specializing in deep document analysis.

You are given:
1. Context passages retrieved from the user's uploaded documents
2. Links to related academic papers found online that are similar to the uploaded paper

Your job is to provide a **comprehensive, detailed answer** using the provided context.

Rules:
1. Give DETAILED, THOROUGH answers. Explain concepts, methodologies, results, and implications in depth.
2. Structure your answer with clear sections using markdown headers (##) when appropriate.
3. Cite sources inline using [Source N] notation.
4. If the paper discusses methodology, explain the approach in detail.
5. If there are results/findings, present them with specifics (numbers, metrics, comparisons).
6. When relevant, explain the significance and real-world applications.
7. At the end, include a "## Related Papers" section listing the similar academic papers found with their titles and links.
8. If context is insufficient, say so but still provide what you can.
9. Write at least 300-500 words for substantive questions.
"""


@traceable(name="Research Assistant RAG")
def run_agent(query: str, chat_history: List[dict]) -> dict:
    """Enhanced RAG pipeline: Retrieve -> Deduplicate -> Rerank -> Extract Topics -> Find Similar Papers -> Generate."""
    
    thoughts = []
    
    # Step 1: Retrieve from FAISS
    thoughts.append("🔎 **Searching** local document index...")
    raw_results = search_faiss(query, k=15)
    
    if not raw_results:
        thoughts.append("⚠️ No documents found in the index.")
        context_text = "No documents have been uploaded yet."
        sources = []
        paper_topic_query = query  # fallback to user query
    else:
        thoughts.append(f"📄 **Retrieved** {len(raw_results)} candidate passages.")
        
        # Step 2: Deduplicate
        deduped = _deduplicate_chunks(raw_results)
        thoughts.append(f"🧹 **Deduplicated** to {len(deduped)} unique passages.")
        
        # Step 3: Rerank
        try:
            thoughts.append("🔀 **Reranking** with Cohere...")
            reranked = rerank(query, deduped, top_n=min(5, len(deduped)))
            thoughts.append(f"✅ **Selected** top {len(reranked)} passages.")
        except Exception as e:
            thoughts.append(f"⚠️ Reranking failed ({e}), using retrieval order.")
            reranked = deduped[:5]
        
        # Build context
        context_parts = []
        sources = []
        for i, doc in enumerate(reranked, 1):
            title = doc.get("title", "Unknown")
            page = doc.get("metadata", {}).get("page", "")
            content = doc.get("content", "")
            score = doc.get("relevance_score", doc.get("score", 0.0))
            
            page_info = f" (Page {page})" if page else ""
            context_parts.append(f"[Source {i}] From: {title}{page_info}\n{content}")
            
            sources.append({
                "title": f"{title}{page_info}",
                "url": doc.get("url", ""),
                "source_type": doc.get("source", doc.get("metadata", {}).get("source", "document")),
                "relevance_score": float(score),
                "content_preview": content[:400],
            })
        
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Step 4: Extract paper topic from context (NOT user query)
        thoughts.append("🏷️ **Extracting** paper topics for finding similar research...")
        try:
            paper_topic_query = _extract_paper_topics(context_text)
            thoughts.append(f"📌 **Topic extracted:** \"{paper_topic_query}\"")
        except Exception as e:
            paper_topic_query = query
            thoughts.append(f"⚠️ Topic extraction failed, using user query.")
    
    # Step 5: Search for SIMILAR PAPERS using extracted topic
    thoughts.append(f"🔬 **Searching ArXiv** for papers similar to: \"{paper_topic_query}\"")
    related_papers = []
    try:
        arxiv_results = search_arxiv(paper_topic_query, max_results=5)
        for paper in arxiv_results:
            related_papers.append({
                "title": paper.get("title", ""),
                "url": paper.get("url", ""),
                "authors": paper.get("authors", ""),
                "source_type": "arxiv",
                "relevance_score": 0.0,
                "content_preview": paper.get("content", "")[:300],
            })
        thoughts.append(f"📚 **Found** {len(related_papers)} similar papers on ArXiv.")
    except Exception as e:
        thoughts.append(f"⚠️ ArXiv search failed: {e}")
    
    # Search web for similar academic papers (using topic, not raw query)
    thoughts.append("🌐 **Searching web** for similar research papers...")
    web_papers = []
    try:
        web_query = f"{paper_topic_query} research paper site:scholar.google.com OR site:arxiv.org OR site:researchgate.net OR site:semanticscholar.org"
        web_results = search_web(web_query)
        for result in web_results[:5]:
            web_papers.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "source_type": "web",
                "relevance_score": 0.0,
                "content_preview": result.get("content", "")[:300],
            })
        thoughts.append(f"🌍 **Found** {len(web_papers)} related papers from the web.")
    except Exception as e:
        thoughts.append(f"⚠️ Web search failed: {e}")
    
    # Step 6: Generate answer
    thoughts.append("🧠 **Generating** detailed answer with Llama-3.3-70b...")
    
    llm = _get_llm()
    
    # Build related papers text for the prompt
    related_text = ""
    all_related = related_papers + web_papers
    if all_related:
        related_text = "\n\n## Similar/Related Academic Papers Found Online:\n"
        for i, paper in enumerate(all_related, 1):
            title = paper.get("title", "Untitled")
            url = paper.get("url", "")
            authors = paper.get("authors", "")
            author_str = f" by {authors}" if authors else ""
            related_text += f"{i}. **{title}**{author_str}\n   Link: {url}\n"
    
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    
    for msg in chat_history[-8:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    
    user_prompt = f"""## Context from uploaded documents:

{context_text}

{related_text}

## User Question:
{query}

Provide a comprehensive, detailed answer. Cite document sources with [Source N]. At the end, list the related papers with their links under a "## Related Papers" heading."""
    
    messages.append(HumanMessage(content=user_prompt))
    
    response = llm.invoke(messages)
    answer = response.content
    thoughts.append("✅ **Answer generated** successfully.")
    
    all_sources = sources + related_papers + web_papers
    
    return {
        "answer": answer,
        "sources": all_sources,
        "intent": "rag_pipeline",
        "raw_chunks": [s.get("content_preview", "") for s in all_sources],
        "thoughts": thoughts,
    }
