# fastapi_app.py
# pip install fastapi uvicorn aiofiles python-multipart pypdf faiss-cpu langchain-google-genai langchain-community google-genai

import os
import asyncio
import logging
import random
from typing import Dict, Optional, List
from uuid import uuid4
from io import BytesIO
import dotenv
dotenv.load_dotenv()
from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

import aiofiles
import numpy as np
from pypdf import PdfReader
import faiss

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from fastapi.staticfiles import StaticFiles

# Gemini streaming
from google import genai
from google.genai import types

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app")

# -----------------------
# App and CORS
# -----------------------
app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # change to your front-end origin in production
#     allow_methods=["*"],
#     allow_headers=["*"],
#     allow_credentials=True,
# )

ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",   # your dev static server
    "http://127.0.0.1:5501",   # your dev static server
    "http://localhost:5500",
    "http://localhost:5173",   # vite default
    "http://localhost:3000",   # react dev server etc
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,   # <- not "*"
    allow_credentials=True,          # must be True to allow cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Registry keyed by session_id (one session per user)
# session structure:
# {
#   "queue": asyncio.Queue,
#   "vs": FAISS | None,
#   "emb": Embeddings | None,
#   "ingestion_task": asyncio.Task | None,
#   "ingestion_done": bool,
#   "ttl_task": asyncio.Task | None
# }
# -----------------------
registry: Dict[str, Dict] = {}

# TTL in seconds
SESSION_TTL = 15 * 60  # 15 minutes

MAX_FILE_BYTES = 200 * 1024 * 1024  # 200MB


# -----------------------
# Utilities: session management
# -----------------------
def make_session_id() -> str:
    return uuid4().hex

def create_session() -> str:
    sid = make_session_id()
    q = asyncio.Queue(maxsize=64)
    registry[sid] = {
        "queue": q,
        "vs": None,
        "emb": None,
        "ingestion_task": None,
        "ingestion_done": False,
        "ttl_task": None,
    }
    logger.info(f"session_created: {sid}")
    # create TTL watcher
    registry[sid]["ttl_task"] = asyncio.create_task(session_ttl_watcher(sid))
    return sid

def get_session_id_from_request(req: Request) -> Optional[str]:
    return req.cookies.get("session_id")

def get_entry(session_id: str) -> Optional[Dict]:
    return registry.get(session_id)

async def safe_put(queue: asyncio.Queue, msg: str):
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        # drop one and enqueue
        try:
            _ = queue.get_nowait()
        except Exception:
            pass
        try:
            queue.put_nowait(msg)
        except Exception:
            pass

async def destroy_session(session_id: str):
    """Destroy the vectorstore and cancel background tasks for a session."""
    entry = registry.get(session_id)
    if not entry:
        return
    logger.info(f"destroy_session: {session_id}")
    # cancel ingestion task if running
    itask = entry.get("ingestion_task")
    if itask and not itask.done():
        try:
            itask.cancel()
        except Exception:
            logger.warning("failed to cancel ingestion task")
    # cancel ttl task
    ttask = entry.get("ttl_task")
    if ttask and not ttask.done():
        try:
            ttask.cancel()
        except Exception:
            pass
    # dereference vs and emb
    entry["vs"] = None
    entry["emb"] = None
    entry["ingestion_task"] = None
    entry["ingestion_done"] = False
    # notify queue and remove
    q: asyncio.Queue = entry.get("queue")
    if q:
        try:
            await safe_put(q, "__SESSION_DESTROYED__")
        except Exception:
            pass
    registry.pop(session_id, None)
    logger.info(f"session_destroyed: {session_id}")

async def reset_session_ttl(session_id: str):
    """Cancel previous TTL task and start a fresh watcher."""
    entry = registry.get(session_id)
    if not entry:
        return
    t = entry.get("ttl_task")
    if t and not t.done():
        try:
            t.cancel()
        except Exception:
            pass
    entry["ttl_task"] = asyncio.create_task(session_ttl_watcher(session_id))

async def session_ttl_watcher(session_id: str):
    """Wait SESSION_TTL seconds, then destroy session."""
    try:
        await asyncio.sleep(SESSION_TTL)
        logger.info(f"session_ttl_expired: {session_id}")
        await destroy_session(session_id)
    except asyncio.CancelledError:
        # TTL was reset/cancelled
        logger.debug(f"ttl_watcher_cancelled: {session_id}")
    except Exception as e:
        logger.exception("ttl_watcher_error")

# -----------------------
# PDF helpers
# -----------------------
def load_pdf_pages_from_bytes(pdf_bytes: bytes) -> List[str]:
    if not pdf_bytes or len(pdf_bytes) < 5:
        raise ValueError("Empty or invalid PDF bytes")
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Uploaded file does not look like a PDF (missing %PDF header)")
    try:
        with BytesIO(pdf_bytes) as bio:
            reader = PdfReader(bio)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    raise ValueError("PDF is encrypted; cannot parse")
            pages = [page.extract_text() or "" for page in reader.pages]
            return pages
    except Exception as e:
        raise ValueError(f"Pdf parsing failed: {e}")

def chunk_pages_to_documents(pages: List[str], pdf_path: str, pages_per_chunk: int, overlap_pages: int = 1):
    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be > 0")
    step = pages_per_chunk - overlap_pages
    if step <= 0:
        raise ValueError("overlap_pages must be smaller than pages_per_chunk")
    chunks = []
    idx = 0
    total = len(pages)
    while idx < total:
        start = idx
        end = min(idx + pages_per_chunk, total)
        text = "\n".join(pages[start:end]).strip()
        if text:
            chunks.append(Document(page_content=text, metadata={"page_range": f"{start+1}-{end}", "source": pdf_path, "title": f"{pdf_path} pages {start+1}-{end}"}))
        idx += step
    return chunks, []

# -----------------------
# Embeddings & FAISS init
# -----------------------
def init_vectorstore_sync(embedding_model: str = "models/gemini-embedding-001"):
    emb = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=os.environ.get("GOOGLE_API_KEY"))
    dim = len(emb.embed_query("probe"))
    index = faiss.IndexFlatL2(dim)
    vs = FAISS(embedding_function=emb, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    return emb, vs

def is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("429" in s) or ("rate" in s and "limit" in s) or ("throttl" in s)

# -----------------------
# Embedding with backoff — sends retry messages via status_put
# -----------------------
async def safe_embed_documents_async(
    emb,
    texts,
    titles,
    status_put,
    task_type="retrieval_document",
    batch_size=None,
    max_retries=6,
    initial_delay=1.0,
    multiplier=2.0,
    max_delay=60.0,
):
    if batch_size is None:
        batch_size = len(texts)
    attempt = 0
    while attempt < max_retries:
        try:
            kwargs = {"batch_size": batch_size}
            if task_type: kwargs["task_type"] = task_type
            if titles is not None: kwargs["titles"] = titles
            vecs = await asyncio.to_thread(emb.embed_documents, texts, **kwargs)
            return vecs
        except Exception as e:
            attempt += 1
            rate = is_rate_limit_error(e)
            delay = min(max_delay, initial_delay * (multiplier ** (attempt - 1)))
            jitter = random.uniform(0, delay * 0.1)
            total_sleep = delay + jitter
            logger.warning(f"embed attempt {attempt}/{max_retries} failed: {e}; sleeping {total_sleep:.1f}s")
            # Only send SSE message for rate-limit/backoff
            if rate and status_put:
                await status_put(f"retrying in {int(total_sleep)}s (attempt {attempt}/{max_retries}) - {str(e)[:200]}")
            if attempt >= max_retries:
                logger.error("embedding failed after retries")
                raise
            await asyncio.sleep(total_sleep)
    raise RuntimeError("unreachable")

# -----------------------
# ingestion task
# -----------------------
async def ingest_documents_with_backoff_async(emb, vs, docs, ids, status_put, batch=5, **backoff_kwargs):
    assert len(docs) == len(ids)
    total = len(docs)
    logger.info(f"ingestion_started: chunks={total} batch={batch}")
    # notify SSE ingestion started
    await status_put("ingestion_started")
    for i in range(0, total, batch):
        batch_docs = docs[i:i+batch]
        texts = [d.page_content for d in batch_docs]
        titles = [d.metadata.get("title") for d in batch_docs]
        vecs = await safe_embed_documents_async(emb, texts, titles, status_put, batch_size=len(texts), **backoff_kwargs)
        vecs_np = np.asarray(vecs, dtype="float32")
        await asyncio.to_thread(vs.index.add, vecs_np)
        base_idx = vs.index.ntotal - len(batch_docs)
        for j, d in enumerate(batch_docs):
            doc_id = ids[i+j]
            vs.docstore.add({doc_id: d})
            vs.index_to_docstore_id[base_idx + j] = doc_id
        logger.info(f"indexed batch {(i//batch)+1}: added {len(batch_docs)} vectors (index size {vs.index.ntotal})")
    logger.info("ingestion_completed")
    await status_put("ingestion_completed")
    # mark ingestion done in registry by caller

# -----------------------
# SSE endpoint: uses session cookie
# -----------------------
@app.get("/events")
async def events(request: Request):
    session_id = get_session_id_from_request(request)
    if not session_id:
        # create session and set cookie
        session_id = create_session()
        # return streaming response but set cookie via header
        entry = get_entry = registry[session_id]
        q = entry["queue"]
        async def event_gen():
            try:
                while True:
                    msg = await q.get()
                    yield f"data: {msg}\n\n"
                    if msg == "__SESSION_DESTROYED__":
                        break
            finally:
                # cleanup session on disconnect
                if registry.get(session_id):
                    # keep session but don't destroy immediately; TTL will handle empties
                    pass
        headers = {
                    "Set-Cookie": (
                        f"session_id={session_id}; Path=/; HttpOnly; SameSite=None; Secure"
                    )
                }
        return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)
    # existing session
    entry = registry.get(session_id)
    if not entry:
        # session expired
        raise HTTPException(status_code=400, detail="Session invalid or expired; refresh to get new session")
    # enforce one listener per session
    q = entry["queue"]
    # If queue already has a connected listener? we don't track connected count; to keep it simple allow single connect by marking a flag:
    if entry.get("_listener_active"):
        raise HTTPException(status_code=409, detail="Listener already connected for this session")
    entry["_listener_active"] = True

    async def event_gen():
        try:
            while True:
                msg = await q.get()
                yield f"data: {msg}\n\n"
                if msg == "__SESSION_DESTROYED__":
                    break
        finally:
            entry["_listener_active"] = False

    return StreamingResponse(event_gen(), media_type="text/event-stream")

# -----------------------
# session endpoint (explicit)
# -----------------------
@app.post("/session")
async def session_endpoint():
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

# -----------------------
# Upload endpoint (no session_id param — uses cookie to associate)
# -----------------------
@app.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    # print(request)
    # print(request.headers)
    print(request.cookies)
    session_id = get_session_id_from_request(request)
    if not session_id or session_id not in registry:
        raise HTTPException(status_code=400, detail="No session; call /session or open /events to get one")
    entry = registry[session_id]
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
    try:
        emb, vs = await asyncio.to_thread(init_vectorstore_sync, "models/gemini-embedding-001")
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
    session_id = get_session_id_from_request(request)
    if not session_id or session_id not in registry:
        raise HTTPException(status_code=400, detail="No session; call /session or open /events to get one")
    entry = registry[session_id]
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
    def gen_stream():
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
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
