"""Session management utilities for the FastAPI application."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional
from uuid import uuid4

from fastapi import Request

logger = logging.getLogger("app.session")

SESSION_TTL = 15 * 60  # 15 minutes

# Registry keyed by session_id. Each entry holds session-scoped state.
registry: Dict[str, Dict] = {}


def make_session_id() -> str:
    """Return a random hex session identifier."""
    return uuid4().hex


def create_session() -> str:
    """Create a new session entry in the registry and schedule TTL handling."""
    sid = make_session_id()
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    registry[sid] = {
        "queue": queue,
        "vs": None,
        "emb": None,
        "ingestion_task": None,
        "ingestion_done": False,
        "ttl_task": None,
        "api_key": None,
    }
    logger.info("session_created: %s", sid)
    registry[sid]["ttl_task"] = asyncio.create_task(session_ttl_watcher(sid))
    return sid


def get_session_id_from_request(req: Request) -> Optional[str]:
    """Extract the session_id cookie from the incoming request, if present."""
    return req.cookies.get("session_id")


def get_entry(session_id: str) -> Optional[Dict]:
    """Return the registry entry for a session, if the session exists."""
    return registry.get(session_id)


async def safe_put(queue: asyncio.Queue, msg: str) -> None:
    """Put a message onto the queue, dropping the oldest item if it is full."""
    try:
        queue.put_nowait(msg)
    except asyncio.QueueFull:
        try:
            _ = queue.get_nowait()
        except Exception:
            pass
        try:
            queue.put_nowait(msg)
        except Exception:
            pass


async def destroy_session(session_id: str) -> None:
    """Tear down session resources and notify listeners that the session ended."""
    entry = registry.get(session_id)
    if not entry:
        return

    logger.info("destroy_session: %s", session_id)
    itask = entry.get("ingestion_task")
    if itask and not itask.done():
        try:
            itask.cancel()
        except Exception:
            logger.warning("failed to cancel ingestion task")

    ttask = entry.get("ttl_task")
    if ttask and not ttask.done():
        try:
            ttask.cancel()
        except Exception:
            pass

    entry["vs"] = None
    entry["emb"] = None
    entry["ingestion_task"] = None
    entry["ingestion_done"] = False

    queue: asyncio.Queue = entry.get("queue")
    if queue:
        try:
            await safe_put(queue, "__SESSION_DESTROYED__")
        except Exception:
            pass

    registry.pop(session_id, None)
    logger.info("session_destroyed: %s", session_id)


async def reset_session_ttl(session_id: str) -> None:
    """Reset the TTL timer for an active session."""
    entry = registry.get(session_id)
    if not entry:
        return

    existing = entry.get("ttl_task")
    if existing and not existing.done():
        try:
            existing.cancel()
        except Exception:
            pass

    entry["ttl_task"] = asyncio.create_task(session_ttl_watcher(session_id))


async def session_ttl_watcher(session_id: str) -> None:
    """Wait for the TTL to expire and destroy the session when it does."""
    try:
        await asyncio.sleep(SESSION_TTL)
        logger.info("session_ttl_expired: %s", session_id)
        await destroy_session(session_id)
    except asyncio.CancelledError:
        logger.debug("ttl_watcher_cancelled: %s", session_id)
    except Exception:
        logger.exception("ttl_watcher_error")
