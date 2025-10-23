# ingest.py
import asyncio
import os
import random
import time
from typing import List, Optional
import numpy as np
from uuid import uuid4
from pypdf import PdfReader
from io import BytesIO

import faiss
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import dotenv

dotenv.load_dotenv()

# --- helpers (similar to your earlier code, but async) ---
print("api key:", os.getenv("GOOGLE_API_KEY"))
def load_pdf_pages_from_bytes(pdf_bytes: bytes) -> List[str]:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return pages

def chunk_pages_to_documents(
    pages: List[str],
    pdf_path: str,
    pages_per_chunk: int,
    overlap_pages: int = 1,
):
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
            chunks.append(
                Document(
                    page_content=text,
                    metadata={"page_range": f"{start_page}-{end_page}", "source": pdf_path, "title": title},
                )
            )
        idx += step
    return chunks, titles

def init_vectorstore_sync(embedding_model: str = "models/gemini-embedding-001"):
    print("api key:", os.getenv("GOOGLE_API_KEY"))
    emb = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=os.getenv("GOOGLE_API_KEY"))
    dim = len(emb.embed_query("probe"))
    index = faiss.IndexFlatL2(dim)
    vs = FAISS(embedding_function=emb, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
    return emb, vs

# ---- Async-wrappers for embedding + backoff notifications ----

def is_rate_limit_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return ("429" in s) or ("rate" in s and "limit" in s) or ("throttl" in s)

async def safe_embed_documents_async(
    emb,
    texts: List[str],
    titles: Optional[List[str]],
    status_put,  # async function to call: await status_put("message")
    task_type: Optional[str] = "retrieval_document",
    batch_size: Optional[int] = None,
    max_retries: int = 6,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
):
    """Run emb.embed_documents in executor with exponential backoff and notify via status_put."""
    if batch_size is None:
        batch_size = len(texts)
    attempt = 0
    while attempt < max_retries:
        try:
            kwargs = {"batch_size": batch_size}
            if task_type is not None:
                kwargs["task_type"] = task_type
            if titles is not None:
                kwargs["titles"] = titles
            # call the blocking embed_documents in a thread so event loop remains responsive
            vecs = await asyncio.to_thread(emb.embed_documents, texts, **kwargs)
            return vecs
        except Exception as e:
            attempt += 1
            is_rate = is_rate_limit_error(e)
            delay = min(max_delay, initial_delay * (multiplier ** (attempt - 1)))
            jitter = random.uniform(0, delay * 0.1)
            total_sleep = delay + jitter
            # notify the client (short message)
            if status_put:
                # be concise and safe for JSON display on client
                short = str(e)[:200]
                await status_put(f"retrying in {int(total_sleep)}s (attempt {attempt}/{max_retries}) - {short}")
            if attempt >= max_retries:
                if status_put:
                    await status_put("error: embedding failed after retries")
                raise
            await asyncio.sleep(total_sleep)
    raise RuntimeError("unreachable")

async def ingest_documents_with_backoff_async(
    emb,
    vs: FAISS,
    docs: List[Document],
    ids: List[str],
    status_put,                 # async callback: await status_put("msg")
    batch: int = 5,
    max_retries: int = 6,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
    task_type: str = "retrieval_document",
):
    assert len(docs) == len(ids)
    total = len(docs)
    for i in range(0, total, batch):
        batch_docs = docs[i:i + batch]
        texts = [d.page_content for d in batch_docs]
        titles = [d.metadata.get("title") for d in batch_docs]
        await status_put(f"ingesting batch {i//batch + 1} ({len(batch_docs)} chunks)...")

        # embed with backoff + status updates
        vecs = await safe_embed_documents_async(
            emb,
            texts,
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
        # Add vectors to FAISS in a thread to avoid blocking
        await asyncio.to_thread(vs.index.add, vecs_np)

        base_idx = vs.index.ntotal - len(batch_docs)
        # Add docs to docstore (cheap operation; do it in event loop)
        for j, d in enumerate(batch_docs):
            doc_id = ids[i + j]
            vs.docstore.add({doc_id: d})
            vs.index_to_docstore_id[base_idx + j] = doc_id

        await status_put(f"added {len(batch_docs)} vectors (index size {vs.index.ntotal})")

    await status_put("__DONE__")   # signal done
