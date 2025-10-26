"""Document ingestion helpers and vector-store utilities."""

from __future__ import annotations

import asyncio
import logging
import random
from io import BytesIO
from typing import List, Optional
import os
import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pypdf import PdfReader
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger("app.ingestion")


def load_pdf_pages_from_bytes(pdf_bytes: bytes) -> List[str]:
    """Extract text from each PDF page provided as raw bytes."""
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
                except Exception as exc:  # pylint: disable=broad-except
                    raise ValueError("PDF is encrypted; cannot parse") from exc
            return [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Pdf parsing failed: {exc}") from exc


def chunk_pages_to_documents(
    pages: List[str],
    pdf_path: str,
    pages_per_chunk: int,
    overlap_pages: int = 1,
) -> tuple[List[Document], List[str]]:
    """Group pages into overlapping chunks, producing LangChain Document objects."""
    if pages_per_chunk <= 0:
        raise ValueError("pages_per_chunk must be > 0")

    step = pages_per_chunk - overlap_pages
    if step <= 0:
        raise ValueError("overlap_pages must be smaller than pages_per_chunk")

    chunks: List[Document] = []
    idx = 0
    total = len(pages)
    while idx < total:
        start = idx
        end = min(idx + pages_per_chunk, total)
        text = "\n".join(pages[start:end]).strip()
        if text:
            metadata = {
                "page_range": f"{start + 1}-{end}",
                "source": pdf_path,
                "title": f"{pdf_path} pages {start + 1}-{end}",
            }
            chunks.append(Document(page_content=text, metadata=metadata))
        idx += step

    return chunks, []


def init_vectorstore_sync(
    embedding_model: str = "models/gemini-embedding-001",
    google_api_key: Optional[str] = None,
) -> tuple[GoogleGenerativeAIEmbeddings, FAISS]:
    """Initialise embeddings and a FAISS index for the ingest pipeline."""
    key = (google_api_key or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise ValueError("No Google API key configured; set GOOGLE_API_KEY or supply one per session.")
    emb = GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=key)
    vs = FAISS(embedding_function=emb, index=None, docstore=InMemoryDocstore({}), index_to_docstore_id={})
    return emb, vs


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect whether an exception likely represents a rate-limit response."""
    lower = str(exc).lower()
    return ("429" in lower) or ("rate" in lower and "limit" in lower) or ("throttl" in lower)


async def safe_embed_documents_async(
    emb,
    texts,
    titles,
    status_put,
    task_type: str = "retrieval_document",
    batch_size: Optional[int] = None,
    max_retries: int = 6,
    initial_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
):
    """Embed a batch of documents with retries and SSE status notifications."""
    if batch_size is None:
        batch_size = len(texts)

    attempt = 0
    while attempt < max_retries:
        try:
            kwargs = {"batch_size": batch_size}
            if task_type:
                kwargs["task_type"] = task_type
            if titles is not None:
                kwargs["titles"] = titles
            vecs = await asyncio.to_thread(emb.embed_documents, texts, **kwargs)
            return vecs
        except Exception as exc:  # pylint: disable=broad-except
            attempt += 1
            rate_limited = is_rate_limit_error(exc)
            delay = min(max_delay, initial_delay * (multiplier ** (attempt - 1)))
            jitter = random.uniform(0, delay * 0.1)
            total_sleep = delay + jitter
            logger.warning(
                "embed attempt %s/%s failed: %s; sleeping %.1fs",
                attempt,
                max_retries,
                exc,
                total_sleep,
            )
            if rate_limited and status_put:
                await status_put(
                    f"retrying in {int(total_sleep)}s (attempt {attempt}/{max_retries}) - {str(exc)[:200]}"
                )
            if attempt >= max_retries:
                logger.error("embedding failed after retries")
                raise
            await asyncio.sleep(total_sleep)

    raise RuntimeError("unreachable")


async def ingest_documents_with_backoff_async(
    emb,
    vs: FAISS,
    docs: List[Document],
    ids: List[str],
    status_put,
    batch: int = 5,
    **backoff_kwargs,
):
    """Embed document batches and add them to FAISS, emitting SSE progress updates."""
    assert len(docs) == len(ids)
    total = len(docs)
    logger.info("ingestion_started: chunks=%s batch=%s", total, batch)
    await status_put("ingestion_started")

    for i in range(0, total, batch):
        batch_docs = docs[i : i + batch]
        texts = [d.page_content for d in batch_docs]
        titles = [d.metadata.get("title") for d in batch_docs]
        vecs = await safe_embed_documents_async(
            emb,
            texts,
            titles,
            status_put,
            batch_size=len(texts),
            **backoff_kwargs,
        )
        vecs_np = np.asarray(vecs, dtype="float32")
        await asyncio.to_thread(vs.index.add, vecs_np)
        base_idx = vs.index.ntotal - len(batch_docs)
        for j, document in enumerate(batch_docs):
            doc_id = ids[i + j]
            vs.docstore.add({doc_id: document})
            vs.index_to_docstore_id[base_idx + j] = doc_id
        logger.info(
            "indexed batch %s: added %s vectors (index size %s)",
            (i // batch) + 1,
            len(batch_docs),
            vs.index.ntotal,
        )

    logger.info("ingestion_completed")
    await status_put("ingestion_completed")
