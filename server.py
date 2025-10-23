# fastapi_app.py
# pip install fastapi uvicorn aiofiles python-multipart

import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from ingest import (
    load_pdf_pages_from_bytes,
    chunk_pages_to_documents,
    init_vectorstore_sync,
    ingest_documents_with_backoff_async,
)
from uuid import uuid4

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change for production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# single listener per file_id: file_id -> asyncio.Queue
listeners: Dict[str, asyncio.Queue] = {}

def create_queue_for_file(file_id: str) -> asyncio.Queue:
    q = asyncio.Queue(maxsize=32)
    listeners[file_id] = q
    return q

def get_queue_for_file(file_id: str):
    return listeners.get(file_id)

async def safe_put(queue: asyncio.Queue, msg: str):
    # small wrapper to avoid blocking when queue full; drops old messages
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        # if full, remove one and put again
        try:
            _ = queue.get_nowait()
        except Exception:
            pass
        try:
            queue.put_nowait(msg)
        except Exception:
            pass

@app.get("/events/{file_id}")
async def events(file_id: str):
    """
    Client must connect here first (SSE) and receive messages for this file_id.
    Only one listener allowed: if a listener exists, return 409.
    """
    if file_id in listeners:
        # only one listener allowed for simplicity
        raise HTTPException(status_code=409, detail="Listener already connected for this file_id")

    q = create_queue_for_file(file_id)

    async def event_generator():
        try:
            while True:
                msg = await q.get()
                yield f"data: {msg}\n\n"
                if msg == "__DONE__":
                    break
        finally:
            # cleanup
            listeners.pop(file_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/upload")
async def upload(file: UploadFile = File(...), file_id: str = None):
    """
    Upload PDF in memory and start ingestion that reports via the queue for file_id.
    Client must have connected to /events/{file_id} before calling this.
    """
    if file_id is None:
        raise HTTPException(status_code=400, detail="file_id is required (connect to /events/{file_id} first)")

    q = get_queue_for_file(file_id)
    if q is None:
        raise HTTPException(status_code=400, detail="No active listener for file_id. Connect to /events/{file_id} first.")

    # basic validation
    if file.content_type != "application/pdf":
        await safe_put(q, "error: only PDF allowed")
        await safe_put(q, "__DONE__")
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    # read bytes (not saved to disk)
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        await safe_put(q, f"error: reading upload failed: {str(e)}")
        await safe_put(q, "__DONE__")
        raise HTTPException(status_code=500, detail="upload read failed")

    # inform client immediately
    await safe_put(q, "file_received")

    # parse pages off the event loop (blocking) using to_thread
    try:
        pages = await asyncio.to_thread(load_pdf_pages_from_bytes, pdf_bytes)
    except Exception as e:
        await safe_put(q, f"error: pdf parsing failed: {str(e)}")
        await safe_put(q, "__DONE__")
        raise HTTPException(status_code=400, detail="PDF parse failed")

    # chunk into docs (cheap)
    pages_per_chunk = 5
    overlap_pages = 1
    docs, titles = chunk_pages_to_documents(pages, "uploaded.pdf", pages_per_chunk, overlap_pages)

    # prepare vectorstore & embeddings (init is blocking; run in thread)
    emb, vs = await asyncio.to_thread(init_vectorstore_sync, "models/gemini-embedding-001")

    # generate ids for docs
    ids = [str(uuid4()) for _ in docs]

    # start ingestion in background (so /upload returns quickly). ingestion will push updates to q.
    asyncio.create_task(
        ingest_documents_with_backoff_async(
            emb=emb,
            vs=vs,
            docs=docs,
            ids=ids,
            status_put=lambda msg: safe_put(q, msg),
            batch=2,
            max_retries=6,
            initial_delay=5,
            multiplier=2,
            max_delay=60,
            task_type="retrieval_document",
        )
    )

    return JSONResponse({"status": "ingestion_started", "file_id": file_id, "chunks": len(docs)})
