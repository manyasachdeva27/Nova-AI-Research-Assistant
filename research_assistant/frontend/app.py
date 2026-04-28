import os
import streamlit as st
import requests
import time

# --- Configuration & Theme ---
st.set_page_config(page_title="Nova | AI Research Assistant", page_icon="🚀", layout="wide")

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Premium Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Outfit:wght@400;600;800&display=swap');

    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .stCaption { color: #94a3b8 !important; font-size: 1rem; }
    
    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.4) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.2rem !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.3);
    }
    
    .stTextInput>div>div>input {
        background-color: rgba(15, 23, 42, 0.8) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .thinking-text {
        animation: pulse 1.5s infinite;
        color: #60a5fa;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🛸 NOVA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Intelligent Research Hub</p>", unsafe_allow_html=True)
    st.markdown("---")

    with st.expander("📂 Knowledge Base", expanded=True):
        uploaded_files = st.file_uploader("Upload PDF sources", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        resp = requests.post(f"{BACKEND_URL}/ingest/pdf", files=files, timeout=300)
                        if resp.status_code == 200:
                            st.success(f"✅ Ingested {resp.json()['chunks_stored']} fragments")
                        else:
                            st.error(f"Error: {resp.json().get('detail', 'Unknown')}")
                    except Exception as e:
                        st.error(f"Network error: {e}")

    with st.expander("🌐 External Data"):
        url_input = st.text_input("Web URL", placeholder="https://...")
        if st.button("Fetch URL", use_container_width=True):
            if url_input:
                with st.spinner("Scraping..."):
                    try:
                        resp = requests.post(f"{BACKEND_URL}/ingest/url", json={"url": url_input})
                        if resp.status_code == 200:
                            st.success(f"✅ Captured {resp.json()['chunks_stored']} fragments")
                        else:
                            st.error(resp.json().get('detail', 'Error'))
                    except Exception as e:
                        st.error(str(e))
        
        st.markdown("---")
        arxiv_query = st.text_input("ArXiv Search", placeholder="Topic...")
        if st.button("Search Papers", use_container_width=True):
            if arxiv_query:
                with st.spinner("Searching ArXiv..."):
                    try:
                        resp = requests.post(f"{BACKEND_URL}/ingest/arxiv", json={"query": arxiv_query, "max_results": 3})
                        if resp.status_code == 200:
                            st.success(f"✅ Harvested {resp.json()['chunks_stored']} fragments")
                        else:
                            st.error(resp.json().get('detail', 'Error'))
                    except Exception as e:
                        st.error(str(e))

    st.markdown("---")
    if st.button("🗑️ Clear Workspace", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --- Main Chat Area ---
st.markdown("<h1 class='main-title'>Nova Research</h1>", unsafe_allow_html=True)
st.caption("RAG Pipeline • Llama-3.3-70b • Groq • Cohere Reranking • ArXiv Discovery")


def render_assistant_msg(msg):
    """Render an assistant message with thinking, answer, and sources."""
    # Thinking process
    if "thoughts" in msg and msg["thoughts"]:
        with st.expander("🧠 Thinking Process", expanded=False):
            for thought in msg["thoughts"]:
                st.markdown(thought)
    
    # Main answer
    st.markdown(msg["content"])
    
    # Split sources into local docs vs. related papers
    all_sources = msg.get("sources", [])
    local_sources = [s for s in all_sources if s.get("source_type") not in ("arxiv", "web")]
    related_papers = [s for s in all_sources if s.get("source_type") in ("arxiv", "web")]
    
    # Show local document sources
    if local_sources:
        with st.expander(f"📄 Document Context ({len(local_sources)} passages)", expanded=False):
            for i, src in enumerate(local_sources, 1):
                title = src.get("title", "Untitled")
                score = src.get("relevance_score", 0.0)
                preview = src.get("content_preview", "")
                
                st.markdown(f"**[{i}] {title}** (relevance: {score:.2f})")
                if preview:
                    st.caption(preview[:250] + "..." if len(preview) > 250 else preview)
                st.markdown("---")
    
    # Show related papers with links
    if related_papers:
        with st.expander(f"🔗 Related Papers & Resources ({len(related_papers)} found)", expanded=False):
            for i, paper in enumerate(related_papers, 1):
                title = paper.get("title", "Untitled")
                url = paper.get("url", "")
                stype = paper.get("source_type", "")
                preview = paper.get("content_preview", "")
                
                badge = "📚 ArXiv" if stype == "arxiv" else "🌐 Web"
                st.markdown(f"**{badge} [{i}] {title}**")
                if url:
                    st.markdown(f"🔗 [{url}]({url})")
                if preview:
                    st.caption(preview[:200] + "..." if len(preview) > 200 else preview)
                st.markdown("---")


# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_assistant_msg(msg)
        else:
            st.markdown(msg["content"])

# Chat Input
user_input = st.chat_input("Ask about your uploaded documents...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.chat_message("assistant"):
        thought_placeholder = st.empty()
        thought_placeholder.markdown(
            "<p class='thinking-text'>🔎 Searching documents, finding related papers, and generating detailed answer...</p>",
            unsafe_allow_html=True
        )
        
        start_time = time.time()
        try:
            history_payload = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history[:-1]
            ]
            resp = requests.post(
                f"{BACKEND_URL}/query",
                json={"question": user_input, "chat_history": history_payload},
                timeout=300
            )
            
            elapsed = time.time() - start_time
            thought_placeholder.empty()

            if resp.status_code == 200:
                data = resp.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                thoughts = data.get("thoughts", [])
                raw_chunks = data.get("raw_chunks", [])
                
                # Attach content_preview from raw_chunks to sources
                for i, src in enumerate(sources):
                    if i < len(raw_chunks) and raw_chunks[i]:
                        src["content_preview"] = raw_chunks[i]
                
                assistant_msg = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "thoughts": thoughts,
                }
                
                render_assistant_msg(assistant_msg)
                st.caption(f"⚡ Generated in {elapsed:.1f}s")
                st.session_state.chat_history.append(assistant_msg)
            else:
                error_msg = resp.json().get("detail", "Server error")
                st.error(f"❌ Backend error: {error_msg}")
        except Exception as e:
            thought_placeholder.empty()
            st.error(f"❌ Connection failed: {e}")
