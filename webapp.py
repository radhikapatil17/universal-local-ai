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

# USING STANDARD GENERATOR INSTEAD OF CHAT FOR VISION RELIABILITY
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
        st.image(img, caption="AI Vision Active", use_column_width=True)
        
        # Convert to Base64 (Ollama requires Base64)
        img_byte_arr = io.BytesIO()
        # Convert to RGB to avoid alpha channel issues in some jpegs/pngs
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
    
    # Indexing Local Files
    indexing = Pipeline()
    indexing.add_component("conv", TextFileToDocument())
    indexing.add_component("writer", DocumentWriter(document_store=document_store))
    indexing.connect("conv", "writer")
    
    local_files = list(Path("my_docs").glob("*.txt"))
    if local_files:
        indexing.run({"conv": {"sources": local_files}})
    
    retriever = InMemoryBM25Retriever(document_store=document_store)
    web_search = DuckduckgoApiWebSearch(top_k=3)
    
    # Standard Prompt Template (No ChatMessages needed for the generate endpoint)
    template = """
Local Docs Context:
{% for doc in documents %} {{ doc.content }} {% endfor %}

Web Search Context:
{% for doc in web_docs %} {{ doc.content }} {% endfor %}

Question: {{ query }}

Instruction: You are a helpful AI. Answer the question using the provided Local Docs first. If they lack the answer, use the Web Search context. If an image is provided, analyze the image to answer the question. If neither docs nor web work, rely on your general knowledge.
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

# Display history
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Input Area
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
                # 1. Fetch Local Docs
                status.update(label="Searching local docs...")
                local_results = retriever.run(query=final_query)["documents"]
                
                # 2. Fetch Web Docs (Conditional)
                web_results = []
                if web_on:
                    status.update(label="Searching the web...")
                    try:
                        web_res = web_search.run(query=final_query)
                        web_results = web_res.get("documents", [])
                        
                        # UI Debugger for Web Search
                        if not web_results:
                            st.warning("⚠️ Web search returned 0 results. (Possible rate limit)")
                        else:
                            with st.expander(f"🌐 Found {len(web_results)} Web Results"):
                                for w in web_results:
                                    st.write(f"- {w.content[:200]}...")
                                    
                    except Exception as e:
                        st.warning(f"Web search failed: {e}")
                
                # 3. Build Prompt
                status.update(label="Building context...")
                prompt_result = prompt_builder.run(
                    documents=local_results,
                    web_docs=web_results,
                    query=final_query
                )
                
                # 4. Generate LLM Output
                status.update(label="Thinking...")
                
                # Format parameters for OllamaGenerator
                llm_kwargs = {"prompt": prompt_result["prompt"]}
                if img_b64:
                    # Ollama API expects images at the root of the generate payload
                    llm_kwargs["generation_kwargs"] = {"images": [img_b64]}
                
                llm_result = llm.run(**llm_kwargs)
                ans = llm_result["replies"][0]
                
                status.update(label="✅ Ready!", state="complete")
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                
            except Exception as e:
                status.update(label="❌ Error occurred", state="error")
                st.error(f"Pipeline Error: {e}")