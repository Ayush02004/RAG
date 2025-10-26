# fastapi_app.py
# pip install fastapi uvicorn aiofiles python-multipart pypdf faiss-cpu langchain-google-genai langchain-community google-genai

import os
import asyncio
import logging
from typing import Optional
from uuid import uuid4

import dotenv
from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel

# Gemini streaming
from google import genai
from google.genai import types

from app.ingestion import (
    chunk_pages_to_documents,
    ingest_documents_with_backoff_async,
    init_vectorstore_sync,
    load_pdf_pages_from_bytes,
)
from app.session import (
    create_session,
    get_entry,
    get_session_id_from_request,
    registry,
    reset_session_ttl,
    safe_put,
)

dotenv.load_dotenv()
# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app")

# -----------------------
# App and CORS
# -----------------------
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # <- not "*"
    allow_credentials=True,          # must be True to allow cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Request models
# -----------------------

class ApiKeyPayload(BaseModel):
    """Payload for storing an optional per-session API key."""

    api_key: Optional[str] = None

MAX_FILE_BYTES = 200 * 1024 * 1024  # 200MB

# -----------------------
# Route implementations
# -----------------------


@app.get("/events")
async def events(request: Request):
    """Stream session-scoped server-sent events, creating a session if needed."""

    session_id = get_session_id_from_request(request)
    created_session = False
    entry = get_entry(session_id) if session_id else None

    if not entry:
        session_id = create_session()
        entry = get_entry(session_id)
        created_session = True

    if entry is None:
        raise HTTPException(status_code=500, detail="Failed to initialize session state")

    if entry.get("_listener_active"):
        raise HTTPException(status_code=409, detail="Listener already connected for this session")

    queue = entry["queue"]
    entry["_listener_active"] = True

    async def event_gen():
        try:
            while True:
                msg = await queue.get()
                yield f"data: {msg}\n\n"
                if msg == "__SESSION_DESTROYED__":
                    break
        finally:
            existing = get_entry(session_id)
            if existing:
                existing["_listener_active"] = False

    headers = None
    if created_session:
        headers = {
            "Set-Cookie": f"session_id={session_id}; Path=/; HttpOnly; SameSite=None; Secure"
        }

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

# -----------------------
# session endpoint (explicit)
# -----------------------
@app.post("/session")
async def session_endpoint():
    """Explicitly create a new session and issue the cookie to the caller."""

    sid = create_session()
    resp = JSONResponse({"session_id": sid})
    resp.set_cookie(
        "session_id",
        sid,
        path="/",
        httponly=True,
        samesite="none",
        secure=True,  # required when SameSite=None
    )
    return resp


@app.post("/api-key")
async def set_api_key(request: Request, payload: ApiKeyPayload):
    """Persist a per-session Gemini API key or fall back to the server default."""

    session_id = get_session_id_from_request(request)
    if not session_id:
        raise HTTPException(status_code=400, detail="No session; call /session or open /events to get one")

    entry = get_entry(session_id)
    if not entry:
        raise HTTPException(status_code=400, detail="Session expired; establish a new session")
    key = (payload.api_key or "").strip()
    using_default = True
    message = "Using default API key from server environment."

    if key:
        entry["api_key"] = key
        using_default = False
        message = "Custom API key saved for this session."
    else:
        entry["api_key"] = None

    reingest_required = bool(entry.get("vs"))

    return JSONResponse(
        {
            "message": message,
            "using_default": using_default,
            "reingest_required": reingest_required,
        }
    )

# -----------------------
# Upload endpoint (no session_id param — uses cookie to associate)
# -----------------------
@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    """Ingest a PDF for the current session and kick off embedding creation."""

    session_id = get_session_id_from_request(request)
    if not session_id:
        raise HTTPException(status_code=400, detail="No session; call /session or open /events to get one")

    entry = get_entry(session_id)
    if not entry:
        raise HTTPException(status_code=400, detail="Session expired; establish a new session")

    q: asyncio.Queue = entry["queue"]

    # file checks
    if file.content_type != "application/pdf":
        await safe_put(q, "error: only PDF allowed")
        logger.error(f"upload_rejected_nonpdf session={session_id} content_type={file.content_type}")
        raise HTTPException(status_code=400, detail="Only PDF allowed")
    try:
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_BYTES:
            await safe_put(q, "error: file too large")
            logger.error(f"upload_too_large session={session_id} size={len(pdf_bytes)}")
            raise HTTPException(status_code=413, detail="File too large")
    except Exception as e:
        await safe_put(q, f"error: reading upload failed: {str(e)}")
        logger.exception("upload read failed")
        raise HTTPException(status_code=500, detail="upload read failed")

    logger.info(f"file_uploaded: session={session_id} size={len(pdf_bytes)}")
    # notify SSE that file uploaded
    await safe_put(q, "file_uploaded")

    # If a previous vectorstore exists for this session, delete it (single-file per session)
    if entry.get("vs") is not None:
        logger.info(f"clearing previous vectorstore for session={session_id}")
        # cancel ingestion task if running
        it = entry.get("ingestion_task")
        if it and not it.done():
            try:
                it.cancel()
            except Exception:
                pass
        # dereference
        entry["vs"] = None
        entry["emb"] = None
        entry["ingestion_done"] = False

    # parse pdf pages (blocking) in thread
    try:
        pages = await asyncio.to_thread(load_pdf_pages_from_bytes, pdf_bytes)
    except ValueError as e:
        await safe_put(q, f"error: pdf parsing failed: {str(e)}")
        await safe_put(q, "__SESSION_DESTROYED__")
        logger.exception("pdf parse failed")
        raise HTTPException(status_code=400, detail=f"PDF parse failed: {str(e)}")

    # chunk
    PAGES_PER_CHUNK = 5
    OVERLAP = 1
    docs, _ = chunk_pages_to_documents(pages, "uploaded.pdf", PAGES_PER_CHUNK, OVERLAP)
    ids = [uuid4().hex for _ in docs]

    # init embeddings+FAISS
    session_api_key = entry.get("api_key")
    try:
        emb, vs = await asyncio.to_thread(init_vectorstore_sync, "models/gemini-embedding-001", session_api_key)
    except Exception as e:
        await safe_put(q, f"error: init embeddings failed: {str(e)}")
        await safe_put(q, "__SESSION_DESTROYED__")
        logger.exception("init_vectorstore failed")
        raise HTTPException(status_code=500, detail="embedding init failed")

    # save to session registry
    entry["vs"] = vs
    entry["emb"] = emb
    entry["ingestion_done"] = False
    # reset TTL
    await reset_session_ttl(session_id)

    # start ingestion in background
    ingestion_task = asyncio.create_task(
        ingest_documents_with_backoff_async(
            emb=emb,
            vs=vs,
            docs=docs,
            ids=ids,
            status_put=lambda m: safe_put(q, m),
            batch=2,
            max_retries=6,
            initial_delay=5,
            multiplier=2,
            max_delay=60,
        )
    )
    entry["ingestion_task"] = ingestion_task

    # ingestion_started message is already sent inside ingestion task; but send again for reliability
    await safe_put(q, "ingestion_started")

    # when ingestion completes, mark done in registry and push final SSE message; we attach a callback
    async def _on_ingest_done(task: asyncio.Task):
        try:
            await task
        except asyncio.CancelledError:
            logger.info("ingestion_task cancelled")
            return
        except Exception:
            logger.exception("ingestion task failed")
            await safe_put(q, "error: ingestion failed")
            return
        entry = registry.get(session_id)
        if entry:
            entry["ingestion_done"] = True
            # ingestion task already pushes 'ingestion_completed', but ensure it's present
            await safe_put(q, "ingestion_completed")
            logger.info(f"ingestion_complete session={session_id}")
            # reset TTL after ingestion so session lasts another TTL period
            await reset_session_ttl(session_id)

    # schedule watcher
    asyncio.create_task(_on_ingest_done(ingestion_task))

    return JSONResponse({"status": "ingestion_started", "session_id": session_id, "chunks": len(docs)})


# -----------------------
# Query endpoint — uses session cookie, only allowed after ingestion_done
# Also sends SSE message 'processing_query' while it runs
# -----------------------
@app.post("/query")
async def query(
    request: Request,
    question: str = Form(...),
    k: int = Form(5),
):
    """Answer a user question by querying the session's FAISS index and streaming tokens."""

    session_id = get_session_id_from_request(request)
    if not session_id:
        raise HTTPException(status_code=400, detail="No session; call /session or open /events to get one")

    entry = get_entry(session_id)
    if not entry:
        raise HTTPException(status_code=400, detail="Session expired; establish a new session")

    q: asyncio.Queue = entry["queue"]

    if not entry.get("ingestion_done", False):
        raise HTTPException(status_code=400, detail="Indexing not complete yet; wait until ingestion_finished")

    vs: FAISS = entry.get("vs")
    emb = entry.get("emb")
    if not vs or not emb:
        raise HTTPException(status_code=500, detail="Vector store unavailable")

    # send SSE processing notification
    await safe_put(q, "processing_query")

    logger.info(f"query_received session={session_id} question={question} k={k}")

    # embed query
    try:
        qvec = await asyncio.to_thread(emb.embed_query, question, task_type="retrieval_query")
    except Exception as e:
        logger.exception("embed_query failed")
        raise HTTPException(status_code=500, detail="query embed failed")

    # search (use thread)
    def search_sync():
        try:
            # prefer similarity_search_with_score
            results = vs.similarity_search_with_score(question, k=k)
            if results and isinstance(results[0], tuple):
                docs = [r[0] for r in results]
            else:
                docs = results
            return docs
        except Exception:
            try:
                hits = vs.similarity_search_by_vector_with_relevance_scores(qvec, k=k)
                return [h[0] for h in hits]
            except Exception as e:
                logger.exception("faiss search failed")
                raise

    try:
        docs = await asyncio.to_thread(search_sync)
    except Exception as e:
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail="search failed")

    context = ""
    for d in docs:
        context += f"page: {d.metadata.get('page_range')} | {d.metadata.get('source')}\n{d.page_content}\n\n"

    logger.info(f"query_context_built session={session_id} len={len(context)}")

    # stream Gemini response
    api_key = entry.get("api_key") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="No API key configured on server")

    def gen_stream():
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash"
        system_text = f"Answer using ONLY the context below. If not available, respond 'I don't know.'\n\n{context}"
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        cfg = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            system_instruction=[types.Part.from_text(text=system_text)],
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ],
        )
        try:
            for chunk in client.models.generate_content_stream(model=model, contents=contents, config=cfg):
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.exception("genai streaming failed")
            yield f"\n\n[error during generation: {str(e)}]"

    # reset TTL because user is active
    await reset_session_ttl(session_id)

    return StreamingResponse(gen_stream(), media_type="text/plain; charset=utf-8")

app.mount("/", StaticFiles(directory="static", html=True), name="static")
