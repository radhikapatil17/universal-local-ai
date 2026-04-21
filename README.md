# 🧠 Universal Local AI

A privacy-first, fully local multimodal AI assistant built with Streamlit, Haystack, and Ollama. This app allows you to chat with your local documents, search the live internet, analyze images and webcam captures, and use voice commands—all running entirely on your local machine without sending your private data to proprietary APIs.

## ✨ Features

- **📄 Local RAG (Retrieval-Augmented Generation):** Drop `.txt` files into the `my_docs` folder and chat with your private data.
- **🌐 Live Web Search Fallback:** Automatically fetches up-to-date information from the internet (via DuckDuckGo) if your local documents don't contain the answer. Features a 20-website deep-dive search context.
- **📸 Multi-modal Vision:** Upload images or take photos directly from your webcam. The AI (`llama3.2-vision`) will analyze the images alongside your text prompts.
- **🎙️ Voice Input:** Speak directly to the AI using the integrated microphone button.
- **🔒 100% Private:** Powered by Ollama, ensuring your documents, photos, and voice data never leave your local machine.

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Orchestration / RAG:** [Haystack 2.0](https://haystack.deepset.ai/)
- **Local LLM & Vision:** [Ollama](https://ollama.com/) (`llama3.2-vision`)
- **Web Search:** DuckDuckGo API
- **Voice Recognition:** Streamlit Mic Recorder

## 📋 Prerequisites

Before running this app, you must have **Ollama** installed on your machine.
1. Download and install [Ollama](https://ollama.com/download).
2. Open your terminal and pull the vision model:
   ```bash
   ollama pull llama3.2-vision

nstallation & Setup
Clone or Download the Repository:
Unzip the project files into a folder (e.g., my_ai_project).

Create a Virtual Environment (Recommended):

Bash
python3 -m venv haystack_env
source haystack_env/bin/activate
Install Dependencies:

Bash
pip install streamlit haystack-ai ollama-haystack duckduckgo-api-haystack streamlit-mic-recorder pillow
Create the Documents Folder:
Create a folder named my_docs in the root directory and add any .txt files you want the AI to read.

Bash
mkdir my_docs
Run the Application:

Bash
streamlit run rag.py
💡 Usage Guide
Asking Questions: Type your query into the chat bar at the bottom or use the 🎙️ mic button to speak.

Web Search: Toggle the "🌐 Enable Internet Search" button in the sidebar. When active, the AI will pull context from up to 20 websites to answer your question.

Image Analysis: Click the "➕ Attach" button next to the chat bar to upload an image or take a photo with your webcam. Ask a question about the image!

Local Docs: Ensure your text files are inside the my_docs folder. The system indexes them automatically on startup. If you add new files while the app is running, click "♻️ Reset System" in the sidebar.

⚠️ Troubleshooting
App freezes on "Thinking..." when using images: Ensure you have downloaded the vision model via Ollama: ollama pull llama3.2-vision.

Web Search returns 0 results: DuckDuckGo may temporarily rate-limit your IP address if you make too many requests too quickly. Wait a few hours or connect to a different network/VPN.

"Bad Interpreter" Error: If you moved the project folder to a new location, your virtual environment path is broken. Delete the haystack_env folder and recreate it using the setup steps above.
