import os
import os
os.environ["USER_AGENT"] = "navigator-app/1.0 (contact: shrutinarwat@gmail.com)"  # Set your own identifier
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
import google.generativeai as genai
import bs4
from agno.agent import Agent
from agno.models.google import Gemini
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from agno.tools.exa import ExaTools

# -----------------------------------------------------------------------------
# NAVIGATOR: Natural Answer Vector Intelligence & Generative Agent for Text Organization & Retrieval
# -----------------------------------------------------------------------------
# NAVIGATOR is a Retrieval-Augmented Generation (RAG) agentic system that helps you
# ingest, search, and synthesize answers from your documents and the web.
# -----------------------------------------------------------------------------

class NavigatorEmbedder(Embeddings):
    """
    Embedding class for NAVIGATOR (Natural Answer Vector Intelligence & Generative Agent for Text Organization & Retrieval)
    Uses Gemini's embedding API.
    """
    def __init__(self, model_name="models/text-embedding-004"):
        genai.configure(api_key=st.session_state.google_api_key)
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )
        return response['embedding']

# Constants
NAVIGATOR_COLLECTION = "navigator-rag-collection"

# Streamlit App Initialization
st.title("🧭 NAVIGATOR: Natural Answer Vector Intelligence & Generative Agent for Text Organization & Retrieval")

# Session State Initialization
defaults = {
    'google_api_key': "",
    'qdrant_api_key': "",
    'qdrant_url': "",
    'navigator_store': None,
    'navigator_sources': [],
    'history': [],
    'exa_api_key': "",
    'use_web_search': False,
    'force_web_search': False,
    'similarity_threshold': 0.7
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sidebar Configuration
st.sidebar.header("🔑 API Configuration")
google_api_key = st.sidebar.text_input("Google API Key", type="password", value=st.session_state.google_api_key)
qdrant_api_key = st.sidebar.text_input("Qdrant API Key", type="password", value=st.session_state.qdrant_api_key)
qdrant_url = st.sidebar.text_input("Qdrant URL", 
                                 placeholder="https://your-cluster.cloud.qdrant.io:6333",
                                 value=st.session_state.qdrant_url)

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.rerun()

st.session_state.google_api_key = google_api_key
st.session_state.qdrant_api_key = qdrant_api_key
st.session_state.qdrant_url = qdrant_url

# Web Search Configuration
st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search)

if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input(
        "Exa AI API Key", 
        type="password",
        value=st.session_state.exa_api_key,
        help="Required for web search fallback when no relevant documents are found"
    )
    st.session_state.exa_api_key = exa_api_key
    
    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input(
        "Custom domains (comma-separated)", 
        value=",".join(default_domains),
        help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org"
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

st.sidebar.header("🎯 Search Configuration")
st.session_state.similarity_threshold = st.sidebar.slider(
    "Document Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    help="Lower values will return more documents but might be less relevant. Higher values are more strict."
)

# NAVIGATOR Utility Functions
def init_navigator_store():
    """Initialize Qdrant client for NAVIGATOR."""
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        return QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=60
        )
    except Exception as e:
        st.error(f"🔴 NAVIGATOR Qdrant connection failed: {str(e)}")
        return None

def process_pdf_navigator(file) -> List:
    """Process PDF file and add NAVIGATOR source metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()
            for doc in documents:
                doc.metadata.update({
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat()
                })
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 NAVIGATOR PDF processing error: {str(e)}")
        return []

def process_web_navigator(url: str) -> List:
    """Process web URL and add NAVIGATOR source metadata."""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "content", "main")
                )
            )
        )
        documents = loader.load()
        for doc in documents:
            doc.metadata.update({
                "source_type": "url",
                "url": url,
                "timestamp": datetime.now().isoformat()
            })
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 NAVIGATOR web processing error: {str(e)}")
        return []

def create_navigator_store(client, texts):
    """Create and initialize NAVIGATOR vector store with documents."""
    try:
        try:
            client.create_collection(
                collection_name=NAVIGATOR_COLLECTION,
                vectors_config=VectorParams(
                    size=768,
                    distance=Distance.COSINE
                )
            )
            st.success(f"📚 NAVIGATOR created new collection: {NAVIGATOR_COLLECTION}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
        navigator_store = QdrantVectorStore(
            client=client,
            collection_name=NAVIGATOR_COLLECTION,
            embedding=NavigatorEmbedder()
        )
        with st.spinner('📤 NAVIGATOR uploading documents to Qdrant...'):
            navigator_store.add_documents(texts)
            st.success("✅ NAVIGATOR documents stored successfully!")
            return navigator_store
    except Exception as e:
        st.error(f"🔴 NAVIGATOR vector store error: {str(e)}")
        return None

# NAVIGATOR Agents
def get_navigator_query_rewriter() -> Agent:
    """NAVIGATOR Query Rewriter Agent."""
    return Agent(
        name="NAVIGATOR Query Rewriter",
        model=Gemini(id="gemini-exp-1206"),
        instructions="""You are NAVIGATOR's query optimizer. 
        1. Analyze the user's question.
        2. Rewrite it to be more precise and search-friendly.
        3. Expand acronyms or technical terms.
        4. Output ONLY the rewritten query.

        Example:
        User: "What does it say about ML?"
        Output: "What are the main concepts and applications of Machine Learning (ML) discussed in the context?"
        """,
        show_tool_calls=False,
        markdown=True,
    )

def get_navigator_web_search() -> Agent:
    """NAVIGATOR Web Search Agent."""
    return Agent(
        name="NAVIGATOR Web Search Agent",
        model=Gemini(id="gemini-exp-1206"),
        tools=[ExaTools(
            api_key=st.session_state.exa_api_key,
            include_domains=search_domains,
            num_results=5
        )],
        instructions="""You are NAVIGATOR's web search expert. Search for relevant information, summarize it, and include sources.""",
        show_tool_calls=True,
        markdown=True,
    )

def get_navigator_rag_agent() -> Agent:
    """NAVIGATOR Main RAG Agent."""
    return Agent(
        name="NAVIGATOR RAG Agent",
        model=Gemini(id="gemini-2.0-flash-thinking-exp-01-21"),
        instructions="""You are NAVIGATOR, the Natural Answer Vector Intelligence & Generative Agent for Text Organization & Retrieval.
- Use provided context from documents or web search to answer accurately and clearly.
- If context is from documents, focus on those.
- If context is from web search, indicate so.
- Always be precise and clear.""",
        show_tool_calls=True,
        markdown=True,
    )

def check_navigator_relevance(query: str, navigator_store, threshold: float = 0.7) -> tuple[bool, List]:
    """Check if NAVIGATOR vector store has relevant docs for the query."""
    if not navigator_store:
        return False, []
    retriever = navigator_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": threshold}
    )
    docs = retriever.invoke(query)
    return bool(docs), docs

# Main NAVIGATOR Application Flow
if st.session_state.google_api_key:
    os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
    genai.configure(api_key=st.session_state.google_api_key)
    
    navigator_client = init_navigator_store()
    
    # File/URL Upload Section
    st.sidebar.header("📁 NAVIGATOR Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter URL")
    
    # Process documents
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.navigator_sources:
            with st.spinner('NAVIGATOR processing PDF...'):
                texts = process_pdf_navigator(uploaded_file)
                if texts and navigator_client:
                    if st.session_state.navigator_store:
                        st.session_state.navigator_store.add_documents(texts)
                    else:
                        st.session_state.navigator_store = create_navigator_store(navigator_client, texts)
                    st.session_state.navigator_sources.append(file_name)
                    st.success(f"✅ NAVIGATOR added PDF: {file_name}")

    if web_url:
        if web_url not in st.session_state.navigator_sources:
            with st.spinner('NAVIGATOR processing URL...'):
                texts = process_web_navigator(web_url)
                if texts and navigator_client:
                    if st.session_state.navigator_store:
                        st.session_state.navigator_store.add_documents(texts)
                    else:
                        st.session_state.navigator_store = create_navigator_store(navigator_client, texts)
                    st.session_state.navigator_sources.append(web_url)
                    st.success(f"✅ NAVIGATOR added URL: {web_url}")

    # Display sources in sidebar
    if st.session_state.navigator_sources:
        st.sidebar.header("📚 NAVIGATOR Sources")
        for source in st.session_state.navigator_sources:
            if source.endswith('.pdf'):
                st.sidebar.text(f"📄 {source}")
            else:
                st.sidebar.text(f"🌐 {source}")

    # Chat Interface
    chat_col, toggle_col = st.columns([0.9, 0.1])

    with chat_col:
        prompt = st.chat_input("Ask NAVIGATOR about your sources...")

    with toggle_col:
        st.session_state.force_web_search = st.toggle('🌐', help="Force NAVIGATOR web search")

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Step 1: NAVIGATOR Query Rewrite
        with st.spinner("🧭 NAVIGATOR optimizing your question..."):
            try:
                query_rewriter = get_navigator_query_rewriter()
                rewritten_query = query_rewriter.run(prompt).content
                with st.expander("🔄 See NAVIGATOR's rewritten query"):
                    st.write(f"Original: {prompt}")
                    st.write(f"Rewritten: {rewritten_query}")
            except Exception as e:
                st.error(f"❌ NAVIGATOR query rewrite error: {str(e)}")
                rewritten_query = prompt

        # Step 2: NAVIGATOR Retrieval or Web Search
        context = ""
        docs = []
        if not st.session_state.force_web_search and st.session_state.navigator_store:
            retriever = st.session_state.navigator_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5, 
                    "score_threshold": st.session_state.similarity_threshold
                }
            )
            docs = retriever.invoke(rewritten_query)
            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                st.info(f"📊 NAVIGATOR found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})")
            elif st.session_state.use_web_search:
                st.info("🔄 NAVIGATOR found no relevant documents, falling back to web search...")

        # Step 3: NAVIGATOR Web Search Fallback
        if (st.session_state.force_web_search or not context) and st.session_state.use_web_search and st.session_state.exa_api_key:
            with st.spinner("🔍 NAVIGATOR searching the web..."):
                try:
                    web_search_agent = get_navigator_web_search()
                    web_results = web_search_agent.run(rewritten_query).content
                    if web_results:
                        context = f"Web Search Results:\n{web_results}"
                        if st.session_state.force_web_search:
                            st.info("ℹ️ NAVIGATOR using web search as requested.")
                        else:
                            st.info("ℹ️ NAVIGATOR using web search as fallback.")
                except Exception as e:
                    st.error(f"❌ NAVIGATOR web search error: {str(e)}")

        # Step 4: NAVIGATOR RAG Response
        with st.spinner("🤖 NAVIGATOR thinking..."):
            try:
                rag_agent = get_navigator_rag_agent()
                if context:
                    full_prompt = f"""Context: {context}

Original Question: {prompt}
Rewritten Question: {rewritten_query}

Please provide a comprehensive answer based on the available information."""
                else:
                    full_prompt = f"Original Question: {prompt}\nRewritten Question: {rewritten_query}"
                    st.info("ℹ️ NAVIGATOR found no relevant information in documents or web search.")

                response = rag_agent.run(full_prompt)
                st.session_state.history.append({
                    "role": "assistant",
                    "content": response.content
                })
                with st.chat_message("assistant"):
                    st.write(response.content)
                    if not st.session_state.force_web_search and 'docs' in locals() and docs:
                        with st.expander("🔍 NAVIGATOR document sources"):
                            for i, doc in enumerate(docs, 1):
                                source_type = doc.metadata.get("source_type", "unknown")
                                source_icon = "📄" if source_type == "pdf" else "🌐"
                                source_name = doc.metadata.get("file_name" if source_type == "pdf" else "url", "unknown")
                                st.write(f"{source_icon} Source {i} from {source_name}:")
                                st.write(f"{doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"❌ NAVIGATOR response error: {str(e)}")

else:
    st.warning("⚠️ Please enter your Google API key in the sidebar to use NAVIGATOR.")