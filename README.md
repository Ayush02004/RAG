# Document Q&A RAG

An end-to-end retrieval augmented generation (RAG) demo that lets you upload a PDF, build a session-scoped FAISS index, and chat with a Gemini-powered assistant about the document in real time.
# Free hosted instance:
```
https://rag-yri6.onrender.com/RAG.html
```
## Features

- Cookie-based session tracking so each browser tab keeps an isolated vector store and API key.
- Async ingestion pipeline that parses PDFs, chunks them, and builds embeddings without blocking the event loop.
- Exponential backoff and retry logic when the Gemini API responds with rate limits.
- Real-time updates via server-sent events (SSE) to drive toasts, progress, and streaming answers in the UI.
- Works with the free tier of the Gemini API—bring your own key and start experimenting immediately.

## Demo

<a href="https://youtu.be/acCM6WsIu0c" target="_blank" rel="noopener">
   <img src="https://img.youtube.com/vi/acCM6WsIu0c/hqdefault.jpg" alt="Document Q&A RAG demo video" width="640">
</a>

## Prerequisites

- **Python 3.9.23** (use this exact interpreter version to avoid compatibility issues)
- A Google API key with access to the Gemini models
- (Recommended) `uv`/`venv`/`conda` or any virtual environment manager

## Initial Setup

1. Create and activate a virtual environment using Python 3.9.23.
2. Install the project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set the Gemini API key in your shell so the backend can authenticate with Google:
   ```bash
   # macOS / Linux
   export GOOGLE_API_KEY="your-google-api-key"

   # Windows PowerShell
   setx GOOGLE_API_KEY "your-google-api-key"
   ```
   > Restart the shell (or your editor) after running `setx` so the environment variable is visible to Python.

## Running the Server

Launch the FastAPI app with Uvicorn:
```bash
uvicorn server:app --host 0.0.0.0 --port 10000
```

Once the server is running, open the frontend in your browser:
```
http://localhost:10000/RAG.html
```

The web UI establishes a session automatically, lets you supply a per-session Gemini API key if desired, and streams answers as tokens arrive from the model.

## Repository Structure

- `server.py` – FastAPI application exposing ingestion, query, and streaming endpoints
- `app/ingestion.py` – PDF parsing, chunking, embedding, and FAISS vector store helpers
- `app/session.py` – Session lifecycle management and SSE coordination
- `static/RAG.html` – Frontend experience for file uploads and chat
- `building_vector_store.py`, `ingest.py`, `requirements.txt`, etc. – supporting scripts and dependency lists

## Troubleshooting

- Ensure the `GOOGLE_API_KEY` environment variable is present before starting Uvicorn; missing credentials will prevent embeddings or completions from initializing.
- If you change the API key while the app is running, re-upload your PDF so the new key is used when rebuilding the vector store.
- Large PDFs can take time to index. Watch the toast notifications and status pill in the UI for progress updates.
- If it hits API rate limits, it will perform retries with exponential backoff so, wait for it.