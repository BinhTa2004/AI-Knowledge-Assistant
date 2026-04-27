# RAG Chat Demo (Streamlit)

A Streamlit-based RAG chatbot using LangChain + Google Gemini. It supports:

- Loading default PDF documents from `../Rag ChatBot/data`
- Uploading additional PDF files from the sidebar
- Chunking + embedding + FAISS retrieval
- Chat-style Q&A interface

## 1) Project Structure

- `app.py`: Streamlit UI
- `ultis/`: RAG core modules (`rag_core.py`, `document_pipeline.py`, `chain_factory.py`, `config.py`)
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container image definition
- `docker-compose.yml`: Docker service setup

## 2) Prerequisites

- Python 3.11+
- A valid `GOOGLE_API_KEY`

Create `.env` in this folder:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## 3) Run Locally

From this folder (`Multi-agent`):

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Then open:

- http://localhost:8501

## 4) Run with Docker Compose

From this folder (`Multi-agent`):

```powershell
docker compose up --build
```

Then open:

- http://localhost:8501

Stop services:

```powershell
docker compose down
```

## 5) Notes

- This setup mounts both:
  - `Multi-agent/data`
  - `../Rag ChatBot/data`
- If `../Rag ChatBot/data` does not exist in your repository, either create it or update the default path in `app.py`.
- Uploaded files are saved to a temporary folder inside the app runtime and then indexed.
- `.dockerignore` is included to keep image builds smaller and faster.

