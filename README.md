# 🧠 RAG Chatbot with Gemini (Streamlit + LangChain)

This project is a Retrieval-Augmented Generation (RAG) chatbot system built with Streamlit, LangChain, Google Gemini, and FAISS. It allows users to interact with their documents through an AI-powered conversational interface.

The system focuses on building a modular and extensible RAG pipeline for document-based question answering.

## 🚀 Key Features

- 📄 Document-based QA using PDF knowledge sources
- 🔎 Semantic retrieval with FAISS vector search
- 🧩 Chunking pipeline for efficient document processing
- 🤖 LLM-powered responses using Google Gemini
- 💬 Interactive chat interface via Streamlit
- 🧱 Modular RAG architecture (pipeline separated from UI)

## ⚙️ Configuration

The application requires a Google Gemini API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

This key is used for generating responses from the LLM.

## 🧠 How It Works (High-Level Flow)

1. Load PDF documents
2. Split documents into semantic chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS vector database
5. User sends a query
6. Retrieve relevant document context
7. Pass context + query to Gemini LLM
8. Return grounded response in chat UI

## 📥 Getting Started (Clone & Run)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the root folder:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run the application

```bash
streamlit run app.py
```

Then open:

- http://localhost:8501

## 🎯 Purpose of the Project

This project demonstrates:

- Real-world implementation of Retrieval-Augmented Generation (RAG)
- Integration of LLMs with private document knowledge
- Clean modular architecture for AI systems
- Practical AI engineering workflow from data to retrieval to generation

It can be extended into:

- Enterprise document assistant
- Research assistant for PDFs
- Multi-source knowledge chatbot

