"""Streamlit app for Sinhala fact-checking system."""
import warnings

# Avoid noisy warnings in the UI logs on Python 3.14+
warnings.filterwarnings(
    "ignore",
    message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
)

import streamlit as st
import threading
from datetime import datetime
# noinspection PyUnresolvedReference
from google import genai
import os
from dotenv import load_dotenv

from src.vector_store import QdrantVectorStore
from src.workflow import FactCheckingWorkflow
from src.cache import SimpleCache
from src.gemini_router import GeminiRouter

# Load environment variables
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
_VECTOR_STORE_LOCK = threading.Lock()

# Initialize session state
@st.cache_resource(show_spinner=False)
def get_vector_store(storage_path: str) -> QdrantVectorStore:
    # Streamlit can execute scripts concurrently across sessions; local Qdrant storage
    # cannot be opened by multiple clients at the same time.
    with _VECTOR_STORE_LOCK:
        return QdrantVectorStore(storage_path)


if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store(QDRANT_PATH)

if "workflow" not in st.session_state:
    st.session_state.workflow = FactCheckingWorkflow(
        st.session_state.vector_store,
        client=client
    )

if "cache" not in st.session_state:
    st.session_state.cache = SimpleCache(ttl_hours=24)

if "router" not in st.session_state:
    st.session_state.router = GeminiRouter(client=client)

if "history" not in st.session_state:
    st.session_state.history = []

# Page configuration
st.set_page_config(
    page_title="සිංහල සත්‍ය සෙවුම්කරු",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 සිංහල සත්‍ය සෙවුම්කරු")
st.subheader("Sinhala Fact-Checking System")

# Sidebar with system info
with st.sidebar:
    st.header("📊 System Status")
    
    # Search quota
    quota = st.session_state.workflow.mcp_client.get_server_status()
    st.subheader("Search Quota")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Tavily",
            f"{quota['tavily']['remaining']}/1000",
            delta=f"-{quota['tavily']['used']}"
        )
    with col2:
        st.metric(
            "Brave",
            f"{quota['brave']['remaining']}/2000",
            delta=f"-{quota['brave']['used']}"
        )
    with col3:
        st.metric("DuckDuckGo", "♾️")
    
    # Cache stats
    st.subheader("Cache")
    cache_stats = st.session_state.cache.get_stats()
    st.metric("Cached Results", cache_stats["size"])
    
    # Gemini router stats
    st.subheader("Gemini Models")
    router_stats = st.session_state.router.get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Flash",
            f"{router_stats['flash']['used']}/15",
            delta=f"remaining: {15-router_stats['flash']['used']}"
        )
    with col2:
        st.metric(
            "Pro",
            f"{router_stats['pro']['used']}/2",
            delta=f"remaining: {2-router_stats['pro']['used']}"
        )
    with col3:
        st.metric(
            "Thinking",
            f"{router_stats['thinking']['used']}/10",
            delta=f"remaining: {10-router_stats['thinking']['used']}"
        )
    
    # Collection stats
    st.subheader("Vector Store")
    politics_stats = st.session_state.vector_store.get_collection_stats("politics")
    economics_stats = st.session_state.vector_store.get_collection_stats("economics")
    health_stats = st.session_state.vector_store.get_collection_stats("health")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Politics", politics_stats["document_count"])
    with col2:
        st.metric("Economics", economics_stats["document_count"])
    with col3:
        st.metric("Health", health_stats["document_count"])

# Main input area
st.divider()
col1, col2 = st.columns([3, 1])

with col1:
    statement = st.text_area(
        "ප්‍රකාශය/ඳුන (Enter a claim in Sinhala or English)",
        height=100,
        placeholder="උදා: ශ්‍රී ලංකාවේ ජිඩීපී එක්සත් ජනපද ඩොලර් බිලියනයි"
    )

with col2:
    st.write("")
    st.write("")
    verify_button = st.button("✓ සත්‍යාපනය", width="stretch")

# Verification logic
if verify_button and statement:
    # Check cache first
    cached_result = st.session_state.cache.get(statement)
    
    if cached_result:
        st.info("🎯 Cache hit - Using cached result")
        result = cached_result
        cached = True
    else:
        # Verify claim
        with st.spinner("පරීක්ෂා කරමින්... (Checking...)"):
            try:
                result = st.session_state.workflow.verify(statement)
                st.session_state.cache.set(statement, result)
                cached = False
            except (RuntimeError, ValueError) as e:
                st.error(f"Error during verification: {str(e)}")
                st.stop()
    
    # Display results
    st.divider()
    
    # Verdict
    verdict = result.get("verdict", "unknown")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if verdict == "true":
            st.success("✅ **සත්‍යයි** (TRUE)")
        elif verdict == "false":
            st.error("❌ **අසත්‍යයි** (FALSE)")
        else:
            st.warning("⚠️ **තොරතුරු ප්‍රමාණවත් නොවේ** (INSUFFICIENT)")

    # Tabs for detailed view
    tab1, tab2, tab3 = st.tabs(["📝 විශ්ලේෂණ (Analysis)", "🧠 කාර්ය ප්‍රවාහය (Thought Process)", "🔍 සාක්ෂි (Evidence)"])

    with tab1:
        st.markdown(result.get("final_report") or result.get("analysis") or "No analysis available")
        
        if result.get("critique"):
            st.info(f"**Reviewer Critique during process:** {result.get('critique')}")

    with tab2:
        st.subheader("Agent Workflow Trace")
        trace = result.get("trace", [])
        if trace:
            st.code(" -> ".join(trace), language="text")
            
            for i, step in enumerate(trace):
                st.text(f"Step {i+1}: {step.capitalize()} Agent active")
        else:
            st.write("No trace information available.")
            
        st.json({
            "domain": result.get("domain"),
            "revision_count": result.get("revision_count"),
            "search_source": result.get("search_source")
        })

    with tab3:
        st.subheader("ලබා ගත් ឩප්‍රකාශ (Retrieved Documents)")
        for i, doc in enumerate(result.get("retrieved_docs", [])[:5]):
            with st.expander(f"{i+1}. {doc.get('source', 'Unknown')} (Score: {doc.get('score', 0):.2f})"):
                st.write(doc.get("text", ""))

        if result.get("search_results"):
             st.subheader("Web Search Results")
             for i, res in enumerate(result.get("search_results")[:5]):
                  st.markdown(f"- [{res.get('title', 'Link')}]({res.get('url', '#')})")
                  st.text(res.get('content', '')[:200] + "...")
    
    # Add to history
    st.session_state.history.append({
        "timestamp": datetime.now(),
        "statement": statement,
        "verdict": verdict,
        "cached": cached
    })

# History section
if st.session_state.history:
    st.divider()
    st.subheader("📚 ඉතිහාසය (History)")
    
    history_df = []
    for item in st.session_state.history[-10:]:  # Last 10
        history_df.append({
            "සમයය": item["timestamp"].strftime("%H:%M:%S"),
            "ප්‍රකාශය": item["statement"][:50] + "...",
            "නිගමනය": item["verdict"],
            "Cache": "✓" if item["cached"] else "✗"
        })
    
    if history_df:
        st.dataframe(history_df, width="stretch")
