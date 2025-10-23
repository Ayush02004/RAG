# fastapi_app.py
# pip install fastapi uvicorn aiofiles python-multipart google-genai langchain-google-genai langchain-community faiss-cpu pypdf

import os
import asyncio
import logging
from typing import Dict, Optional
from uuid import uuid4
import dotenv
dotenv.load_dotenv()
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

# AI libs (make sure GOOGLE_API_KEY set in env)
import numpy as np
from pypdf import PdfReader
import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Gemini Streaming
from google import genai
from google.genai import types

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest_app")

# -----------------------
# App + CORS
# -----------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # use your front-end origin in production
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# -----------------------
# In-memory registry:
# file_id -> {'queue': asyncio.Queue, 'vs': FAISS or None, 'emb': Embeddings or None}
# -----------------------
registry: Dict[str, Dict] = {}

MAX_FILE_BYTES = 200 * 1024 * 1024  # 200 MB

# -----------------------
# Helpers: PDF load & chunking (synchronous blocking; we call via to_thread)
# -----------------------
# put at top of file
from io import BytesIO
from typing import List
from pypdf import PdfReader

def load_pdf_pages_from_bytes(pdf_bytes: bytes) -> List[str]:
    """
    Safely load PDF pages from raw bytes and return list[str] (one element per page).
    Raises a ValueError if bytes don't look like a PDF or PdfReader fails.
    """
    if not pdf_bytes or len(pdf_bytes) < 5:
        raise ValueError("Empty or too-small PDF bytes")

    # quick sanity check for PDF header
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Uploaded file does not look like a PDF (missing %PDF header)")

    try:
        with BytesIO(pdf_bytes) as bio:
            reader = PdfReader(bio)
            # handle encrypted PDFs (try empty password)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")  # try empty password
                except Exception:
                    raise ValueError("PDF is encrypted; cannot parse")

            pages = [page.extract_text() or "" for page in reader.pages]
            return pages
    except Exception as e:
        # raise a clear ValueError so caller can catch and signal a friendly error
        raise ValueError(f"Pdf parsing failed: {e}")


def chunk_pages_to_documents(pages, pdf_path: str, pages_per_chunk: int, overlap_pages: int = 1):
    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be > 0")
    if overlap_pages < 0:
        raise ValueError("overlap_pages must be >= 0")
    step = pages_per_chunk - overlap_pages
    if step <= 0:
        raise ValueError("overlap_pages must be smaller than pages_per_chunk")

    chunks = []
    titles = []
    idx = 0
    total_pages = len(pages)
    while idx < total_pages:
        start_idx = idx
        end_idx = min(idx + pages_per_chunk, total_pages)
        start_page = start_idx + 1
        end_page = end_idx
        text = "\n".join(pages[start_idx:end_idx]).strip()
        if text:
            title = f"{pdf_path} pages {start_page}-{end_page}"
            titles.append(title)
            chunks.append(Document(page_content=text, metadata={"page_range": f"{start_page}-{end_page}", "source": pdf_path, "title": title}))
        idx += step
    return chunks, titles

# -----------------------
# Embedding + FAISS init (blocking, run in thread)
# -----------------------
def init_vectorstore_sync(embedding_model: str = "models/gemini-embedding-001"):
    emb = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=os.getenv("GOOGLE_API_KEY"))
    dim = len(emb.embed_query("probe"))
    index = faiss.IndexFlatL2(dim)
    vs = FAISS(embedding_function=emb, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    return emb, vs

# -----------------------
# Helper to create registry entry
# -----------------------
def create_registry_entry(file_id: str):
    q = asyncio.Queue(maxsize=64)
    registry[file_id] = {"queue": q, "vs": None, "emb": None}
    return registry[file_id]

def get_registry_entry(file_id: str):
    return registry.get(file_id)

async def safe_put(queue: asyncio.Queue, msg: str):
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        # drop oldest and push new
        try:
            _ = queue.get_nowait()
        except Exception:
            pass
        try:
            queue.put_nowait(msg)
        except Exception:
            pass

# -----------------------
# is_rate helper
# -----------------------
def is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("429" in s) or ("rate" in s and "limit" in s) or ("throttl" in s)

# -----------------------
# safe embed (runs embedding in thread, sends only retry messages via status_put)
# -----------------------
import random, time
async def safe_embed_documents_async(
    emb,
    texts,
    titles,
    status_put,  # async callable: await status_put(msg)
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
            # embed in thread (blocking)
            vecs = await asyncio.to_thread(emb.embed_documents, texts, **kwargs)
            return vecs
        except Exception as e:
            attempt += 1
            rate = is_rate_limit_error(e)
            delay = min(max_delay, initial_delay * (multiplier ** (attempt - 1)))
            jitter = random.uniform(0, delay * 0.1)
            total_sleep = delay + jitter
            # LOG always
            logger.warning(f"embed_documents attempt {attempt}/{max_retries} failed: {e}; sleeping {total_sleep:.1f}s")
            # Only send SSE messages when it's a rate-limit/backoff situation (per your request)
            if rate and status_put:
                # concise message
                short = str(e)[:200]
                await status_put(f"retrying in {int(total_sleep)}s (attempt {attempt}/{max_retries}) - {short}")
            if attempt >= max_retries:
                logger.error("embedding failed after retries")
                raise
            await asyncio.sleep(total_sleep)
    raise RuntimeError("unreachable")

# -----------------------
# ingestion background task (calls safe_embed_documents_async)
# -----------------------
async def ingest_documents_with_backoff_async(
    emb,
    vs,
    docs,
    ids,
    status_put,  # async callable
    batch=5,
    max_retries=6,
    initial_delay=1.0,
    multiplier=2.0,
    max_delay=60.0,
    task_type="retrieval_document",
):
    assert len(docs) == len(ids)
    total = len(docs)
    logger.info(f"Start ingestion: {total} chunks, batch={batch}")
    for i in range(0, total, batch):
        batch_docs = docs[i:i+batch]
        texts = [d.page_content for d in batch_docs]
        titles = [d.metadata.get("title") for d in batch_docs]
        # Note: per your request we do NOT broadcast normal progress messages over SSE,
        # only retry notifications inside safe_embed_documents_async will be sent via status_put.

        vecs = await safe_embed_documents_async(
            emb=emb,
            texts=texts,
            titles=titles,
            status_put=status_put,
            task_type=task_type,
            batch_size=len(texts),
            max_retries=max_retries,
            initial_delay=initial_delay,
            multiplier=multiplier,
            max_delay=max_delay,
        )

        vecs_np = np.asarray(vecs, dtype="float32")
        # add vectors to FAISS in thread
        await asyncio.to_thread(vs.index.add, vecs_np)
        base_idx = vs.index.ntotal - len(batch_docs)
        for j, d in enumerate(batch_docs):
            doc_id = ids[i+j]
            vs.docstore.add({doc_id: d})
            vs.index_to_docstore_id[base_idx + j] = doc_id

        logger.info(f"Indexed batch {i//batch + 1}: added {len(batch_docs)} vectors (index size {vs.index.ntotal})")
    logger.info("Ingestion complete")

# -----------------------
# SSE endpoint (one listener per file_id)
# -----------------------
@app.get("/events/{file_id}")
async def events(file_id: str):
    if get_registry_entry(file_id) is not None:
        # block multiple listeners - minimal approach
        raise HTTPException(status_code=409, detail="Listener already connected for this file_id")

    entry = create_registry_entry(file_id)
    q = entry["queue"]

    async def event_generator():
        try:
            while True:
                msg = await q.get()
                if msg is None:
                    continue
                yield f"data: {msg}\n\n"
                # we do not send completion markers by default; it's optional.
                # If a component wants to notify completion it can push "__DONE__" and client can close.
                if msg == "__DONE__":
                    break
        finally:
            # cleanup
            registry.pop(file_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -----------------------
# Upload endpoint: read pdf into memory, start ingestion background task and store vs/emb in registry
# -----------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...), file_id: Optional[str] = Form(None)):
    if file_id is None:
        raise HTTPException(status_code=400, detail="file_id is required. Connect to /events/{file_id} first")

    entry = get_registry_entry(file_id)
    if entry is None:
        raise HTTPException(status_code=400, detail="No active SSE listener for this file_id. Connect to /events/{file_id} first")

    q = entry["queue"]

    # validate
    if file.content_type != "application/pdf":
        # only retry messages go out via SSE — we will log and send an error message as SSE as well (it's allowed)
        await safe_put(q, "error: only PDF allowed")
        await safe_put(q, "__DONE__")
        logger.error(f"Upload rejected: non-pdf content_type={file.content_type} for file_id={file_id}")
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    # read bytes
    try:
        pdf_bytes = await file.read()
        if len(pdf_bytes) > MAX_FILE_BYTES:
            await safe_put(q, "error: file too large")
            await safe_put(q, "__DONE__")
            logger.error(f"Upload too large for file_id={file_id} size={len(pdf_bytes)}")
            raise HTTPException(status_code=413, detail="File too large")
    except Exception as e:
        await safe_put(q, f"error: reading upload failed: {str(e)}")
        await safe_put(q, "__DONE__")
        logger.exception("upload read failed")
        raise HTTPException(status_code=500, detail="upload read failed")

    logger.info(f"file_received: file_id={file_id} size={len(pdf_bytes)}")
    # NOTE: per your request, we do NOT broadcast file_received to SSE except for retry messages. Logging is present.

    # parse pages (blocking) in a thread
    try:
        pages = await asyncio.to_thread(load_pdf_pages_from_bytes, pdf_bytes)
    except Exception as e:
        await safe_put(q, f"error: pdf parsing failed: {str(e)}")
        await safe_put(q, "__DONE__")
        logger.exception("pdf parse failed")
        raise HTTPException(status_code=400, detail="PDF parse failed")

    # chunk
    PAGES_PER_CHUNK = 5
    OVERLAP = 1
    docs, titles = chunk_pages_to_documents(pages, "uploaded.pdf", PAGES_PER_CHUNK, OVERLAP)
    ids = [str(uuid4()) for _ in docs]

    # init embeddings + FAISS (blocking) in thread
    try:
        emb, vs = await asyncio.to_thread(init_vectorstore_sync, "models/gemini-embedding-001")
    except Exception as e:
        await safe_put(q, f"error: init embeddings failed: {str(e)}")
        await safe_put(q, "__DONE__")
        logger.exception("init_vectorstore failed")
        raise HTTPException(status_code=500, detail="embedding init failed")

    # store vs & emb in registry so /query can access it
    entry["vs"] = vs
    entry["emb"] = emb

    logger.info(f"indexing_start: file_id={file_id} chunks={len(docs)}")

    # start ingestion in background — ingestion will send only retry messages via safe_put
    asyncio.create_task(ingest_documents_with_backoff_async(
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
    ))

    return JSONResponse({"status": "ingestion_started", "file_id": file_id, "chunks": len(docs)})

# -----------------------
# Utility to combine docs -> context
# -----------------------
def combine_docs_to_context(docs):
    context = ""
    for i, d in enumerate(docs):
        context += f"page: {d.metadata.get('page_range')} | {d.metadata.get('source')}\n"
        context += f"{d.page_content}\n\n"
    return context

# -----------------------
# QUERY endpoint: runs retrieval on in-memory vs for given file_id and streams Gemini response
# returns a streaming text/plain response (chunks appended as generated)
# -----------------------
@app.post("/query")
async def query(file_id: str = Form(...), question: str = Form(...), k: int = Form(5)):
    entry = get_registry_entry(file_id)
    if entry is None or entry.get("vs") is None or entry.get("emb") is None:
        raise HTTPException(status_code=400, detail="No index available for this file_id yet (or ingestion not started)")

    vs: FAISS = entry["vs"]
    emb = entry["emb"]

    logger.info(f"query_received: file_id={file_id} question={question} k={k}")

    # 1) embed the query (blocking) in thread
    try:
        qvec = await asyncio.to_thread(emb.embed_query, question, task_type="retrieval_query")
    except Exception as e:
        logger.exception("embed_query failed")
        raise HTTPException(status_code=500, detail="query embed failed")

    # 2) run similarity search using FAISS in thread (returns documents)
    def search_sync():
        # use wrapper that accepts raw vector if available
        try:
            results = vs.similarity_search_with_score(question, k=k)  # fallback if embed inside
            # results is list of (Document, score) OR list of Document depending on version
            if results and isinstance(results[0], tuple):
                docs = [r[0] for r in results]
            else:
                docs = results
            return docs
        except Exception:
            # fallback: try similarity_search_by_vector_with_relevance_scores
            try:
                hits = vs.similarity_search_by_vector_with_relevance_scores(qvec, k=k)
                # hits = list of (Document, score)
                return [h[0] for h in hits]
            except Exception as e:
                logger.exception("faiss search failed")
                raise

    try:
        docs = await asyncio.to_thread(search_sync)
    except Exception as e:
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail="search failed")

    context = combine_docs_to_context(docs)
    logger.info(f"query_context_built: file_id={file_id} context_len={len(context)}")

    # 3) Stream responses from Gemini model via genai streaming API (sync iterator)
    def gen_stream():
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = "gemini-2.5-flash"
        # Build system instruction with context (keep it concise to avoid token explosion)
        system_text = f"Answer the question using only the provided context. Context follows:\n{context}\n---\nIf context does not contain the answer, respond with 'I don't know.'"
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
            # stream and yield text chunks
            for chunk in client.models.generate_content_stream(model=model, contents=contents, config=cfg):
                # chunk.text may be None for some event types; handle
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.exception("genai streaming failed")
            yield f"\n\n[error during generation: {str(e)}]"
    # Return as streaming plain text (front-end will append chunks)
    return StreamingResponse(gen_stream(), media_type="text/plain; charset=utf-8")
