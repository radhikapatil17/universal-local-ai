import streamlit as st
import os
import io
import base64
from pathlib import Path
from PIL import Image

# Haystack & Ollama Imports
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters.txt import TextFileToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever

from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator
from duckduckgo_api_haystack import DuckduckgoApiWebSearch
from streamlit_mic_recorder import speech_to_text

# --- 1. PAGE CONFIG & MODERN UI ---
st.set_page_config(page_title="Universal AI", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .stApp { background: #0e1117; color: white; }
    .stChatMessage { border-radius: 15px; border: 1px solid #30363d; background: #161b22; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #238636; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR (Photos & Docs) ---
with st.sidebar:
    st.title("📂 Workspace")
    
    st.subheader("📸 Image Analysis")
    uploaded_image = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
    img_b64 = None
    
    if uploaded_image:
        img = Image.open(uploaded_image)
        # Fixed the Streamlit warning here by using use_container_width
        st.image(img, caption="AI Vision Active", use_container_width=True)
        
        img_byte_arr = io.BytesIO()
        img = img.convert('RGB')
        img.save(img_byte_arr, format='JPEG')
        img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    st.divider()
    web_on = st.toggle("🌐 Enable Internet Search", value=True)
    
    st.subheader("📄 Local Knowledge")
    docs_path = Path("my_docs")
    if not docs_path.exists(): docs_path.mkdir()
    files = [f.name for f in docs_path.glob("*.txt")]
    for f in files: st.caption(f"✅ {f}")
    
    if st.button("♻️ Reset System"):
        st.cache_resource.clear()
        st.rerun()

# --- 3. MODULAR RAG ENGINE ---
@st.cache_resource
def setup_components():
    document_store = InMemoryDocumentStore()
    
    indexing = Pipeline()
    indexing.add_component("conv", TextFileToDocument())
    indexing.add_component("writer", DocumentWriter(document_store=document_store))
    indexing.connect("conv", "writer")
    
    local_files = list(Path("my_docs").glob("*.txt"))
    if local_files:
        indexing.run({"conv": {"sources": local_files}})
    
    retriever = InMemoryBM25Retriever(document_store=document_store)
    
    # Bumping up to 20 results for maximum safe breadth
    web_search = DuckduckgoApiWebSearch(top_k=20)
    
    # STRICTER PROMPT: Forcing the AI to use the text
    template = """
You are a highly intelligent and thorough research assistant.

LOCAL DOCUMENTS:
{% for doc in documents %}
- {{ doc.content }}
{% endfor %}

WEB SEARCH RESULTS:
{% for doc in web_docs %}
- {{ doc.content }}
{% endfor %}

USER QUESTION: {{ query }}

CRITICAL INSTRUCTIONS: 
1. You MUST read the Web Search Results and Local Documents provided above.
2. Extract as much relevant information from that text as possible to answer the User Question.
3. If an image is provided, analyze it carefully.
4. Do not make up information. If the answer is not in the text or image, state what you know generally, but clarify that it wasn't in the provided search results.
"""
    
    prompt_builder = PromptBuilder(template=template)
    llm = OllamaGenerator(model="llama3.2-vision", url="http://localhost:11434")

    return retriever, web_search, prompt_builder, llm

retriever, web_search, prompt_builder, llm = setup_components()

# --- 4. CHAT UI ---
st.title("🧠 Universal Local AI")
st.write("Talk to your docs, the web, or analyze photos—all privately on your Mac.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

col1, col2 = st.columns([1, 12])
with col1:
    voice_input = speech_to_text(language='en', start_prompt="🎙️", stop_prompt="⏹️", key='STT')

prompt = st.chat_input("Ask a question...")
final_query = voice_input if voice_input else prompt

if final_query:
    st.session_state.messages.append({"role": "user", "content": final_query})
    with st.chat_message("user"): st.markdown(final_query)

    with st.chat_message("assistant"):
        with st.status("🔍 Processing...", expanded=True) as status:
            try:
                status.update(label="Searching local docs...")
                local_results = retriever.run(query=final_query)["documents"]
                
                web_results = []
                if web_on:
                    status.update(label="Searching the web globally...")
                    try:
                        web_res = web_search.run(query=final_query)
                        web_results = web_res.get("documents", [])
                        
                        if not web_results:
                            st.warning("⚠️ Web search returned 0 results.")
                        else:
                            with st.expander(f"🌐 Extracted Data from {len(web_results)} Websites"):
                                for w in web_results:
                                    st.write(f"- {w.content[:200]}...")
                                    
                    except Exception as e:
                        st.warning(f"Web search failed: {e}")
                
                status.update(label="Building context...")
                prompt_result = prompt_builder.run(
                    documents=local_results,
                    web_docs=web_results,
                    query=final_query
                )
                
                status.update(label="Thinking (Expanded Memory Active)...")
                
                # CRITICAL FIX: Expanding the context window (num_ctx) so the AI doesn't forget the text
                llm_kwargs = {
                    "prompt": prompt_result["prompt"],
                    "generation_kwargs": {
                        "num_ctx": 8192,  # Expands memory to handle 20 websites
                        "temperature": 0.3 # Lowers hallucination, forces it to stick to the text
                    }
                }
                
                if img_b64:
                    llm_kwargs["generation_kwargs"]["images"] = [img_b64]
                
                llm_result = llm.run(**llm_kwargs)
                ans = llm_result["replies"][0]
                
                status.update(label="✅ Ready!", state="complete")
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                
            except Exception as e:
                status.update(label="❌ Error occurred", state="error")
                st.error(f"Pipeline Error: {e}")