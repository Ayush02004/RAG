# fastapi_app.py
# pip install fastapi uvicorn aiofiles python-multipart

import os
import uuid
import asyncio
from typing import Dict, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# Allow your frontend origin (change for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use explicit origin in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Mapping file_id -> list of asyncio.Queue objects (one queue per connected SSE client)
listeners: Dict[str, List[asyncio.Queue]] = {}

# Limits
MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB


def ensure_listeners(file_id: str) -> List[asyncio.Queue]:
    """Return the listener list for file_id, creating it if needed."""
    if file_id not in listeners:
        listeners[file_id] = []
    return listeners[file_id]


async def broadcast(file_id: str, message: str) -> None:
    """Broadcast message to all SSE listeners for a file_id."""
    queues = listeners.get(file_id)
    if not queues:
        return
    # Put message into all queues; ignore queues that are full/closed
    to_remove = []
    for q in queues:
        try:
            # Use put_nowait so we don't block; drop messages if queue full
            q.put_nowait(message)
        except asyncio.QueueFull:
            # If full, skip (or optionally remove)
            pass
        except Exception:
            to_remove.append(q)
    # cleanup failed queues
    for q in to_remove:
        try:
            queues.remove(q)
        except ValueError:
            pass
    # remove empty list entry
    if not queues:
        listeners.pop(file_id, None)


async def save_upload_file_streaming(upload_file: UploadFile, dest_path: str, max_bytes: int | None = None, file_id: str | None = None):
    """Stream UploadFile to disk in chunks, enforce max size, broadcast progress minimal messages."""
    total = 0
    chunk_size = 1024 * 1024  # 1 MB chunk writes
    try:
        async with aiofiles.open(dest_path, "wb") as out:
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    # cleanup partial file
                    await out.close()
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass
                    # broadcast error
                    if file_id:
                        await broadcast(file_id, f"error: file too large (>{max_bytes} bytes)")
                        await broadcast(file_id, "__DONE__")
                    raise HTTPException(status_code=413, detail="File too large")
                await out.write(chunk)
    finally:
        try:
            await upload_file.close()
        except Exception:
            pass


@app.post("/upload")
async def upload(file: UploadFile = File(...), file_id: str = Form(...)):
    """
    Upload a PDF and save to disk. Expects form fields:
    - file: the PDF file
    - file_id: the client-generated ID used for SSE
    """
    # Basic checks
    if file.content_type != "application/pdf":
        await broadcast(file_id, "error: only PDF allowed")
        await broadcast(file_id, "__DONE__")
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Ensure listeners structure exists so SSE can be picked up
    ensure_listeners(file_id)

    # Announce that upload has been received (server has accepted POST)
    await broadcast(file_id, "file_received")

    # Create unique storage filename to avoid collisions
    safe_name = f"{uuid.uuid4().hex}.pdf"
    dest_path = os.path.join(UPLOAD_DIR, safe_name)

    # Stream-save the file (async) - this happens inside the request handler,
    # streaming avoids buffering entire file in memory.
    try:
        await save_upload_file_streaming(file, dest_path, max_bytes=MAX_FILE_BYTES, file_id=file_id)
    except HTTPException as he:
        # If the streaming function raised (e.g., 413), the error was already broadcast above
        raise he
    except Exception as exc:
        await broadcast(file_id, f"error: saving failed: {str(exc)}")
        await broadcast(file_id, "__DONE__")
        raise HTTPException(status_code=500, detail="Saving failed") from exc

    # Broadcast that file is saved
    await broadcast(file_id, f"file_saved:{safe_name}")
    # Optionally broadcast a special done marker so client can close the stream
    await broadcast(file_id, "__DONE__")

    # Return immediate JSON with stored filename
    return JSONResponse({"file_id": file_id, "stored_name": safe_name, "status": "saved"})


@app.get("/events/{file_id}")
async def events(file_id: str):
    """
    SSE endpoint. Clients should connect to /events/{file_id} before uploading.
    Each connected client gets its own small queue to receive messages.
    """
    # Create queue and register it
    q: asyncio.Queue = asyncio.Queue(maxsize=16)  # small bounded queue to avoid memory growth
    ensure_listeners(file_id).append(q)

    async def event_generator():
        try:
            while True:
                # Wait for next message
                msg = await q.get()
                if msg is None:
                    continue
                # SSE format: data: <message>\n\n
                yield f"data: {msg}\n\n"
                # If a done marker is received, finish this generator
                if msg == "__DONE__":
                    break
        finally:
            # Remove this queue from listeners on disconnect
            lst = listeners.get(file_id)
            if lst and q in lst:
                try:
                    lst.remove(q)
                except ValueError:
                    pass
            # Clean up empty listener lists
            if not listeners.get(file_id):
                listeners.pop(file_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
