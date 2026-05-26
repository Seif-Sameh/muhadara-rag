"""Muhadara RAG — FastAPI server.

Serves a custom HTML frontend + REST API. Replaces the previous Gradio-based app.
The backend (utils.py) is unchanged — Gradio was only the UI layer.

Endpoints:
    GET  /                  static index.html
    GET  /static/*          assets (JS, CSS, demo.mp3, etc.)
    POST /api/ask           RAG over the pre-indexed demo lecture (Qdrant + Groq)
    POST /api/upload        multipart upload → ASR on Modal → in-memory index
    POST /api/ask-upload    RAG over a session-specific in-memory store
    GET  /api/health        liveness probe
"""
from __future__ import annotations

import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from utils import (
    E5Embeddings,
    MAX_UPLOAD_SECONDS,
    SimpleVectorStore,
    build_chains,
    chunk_segments,
    format_context,
    format_timestamp,
    load_vector_stores,
    make_qdrant_retriever,
    transcribe,
)

# ── Config (Space secrets) ───────────────────────────────────
QDRANT_URL        = os.environ["QDRANT_URL"]
QDRANT_API_KEY    = os.environ["QDRANT_API_KEY"]
GROQ_API_KEY      = os.environ["GROQ_API_KEY"]
DEMO_LECTURE_STEM = "lec1"
ROOT              = Path(__file__).parent
STATIC_DIR        = ROOT / "static"


# ── Global runtime state ─────────────────────────────────────
class Runtime:
    embeddings: E5Embeddings | None = None
    rag_inner = None
    summary_chain = None
    qdrant_retrieve = None
    # Per-session in-memory stores for uploaded recordings. UUID → SimpleVectorStore.
    # Lives only for the lifetime of the process; fine for a demo, no cleanup needed.
    sessions: dict[str, SimpleVectorStore] = {}


rt = Runtime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    print("⏳ Loading embeddings …", flush=True)
    rt.embeddings = E5Embeddings()
    print("⏳ Connecting Qdrant …", flush=True)
    per, glob = load_vector_stores(rt.embeddings, DEMO_LECTURE_STEM, QDRANT_URL, QDRANT_API_KEY)
    print("⏳ Building chains …", flush=True)
    rt.rag_inner, rt.summary_chain = build_chains(GROQ_API_KEY)
    rt.qdrant_retrieve = make_qdrant_retriever(per, glob, rt.embeddings)
    print("✅ Ready", flush=True)
    yield


app = FastAPI(title="Muhadara RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Schemas ──────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AskUploadRequest(BaseModel):
    session_id: str
    question: str = Field(..., min_length=1, max_length=2000)


# ── Helpers ──────────────────────────────────────────────────
def _serialize_hits(hits) -> list[dict[str, Any]]:
    return [
        {
            "text":      doc.page_content,
            "abs_start": float(doc.metadata.get("abs_start", 0)),
            "timestamp": format_timestamp(float(doc.metadata.get("abs_start", 0))),
            "score":     float(score),
        }
        for doc, score in hits
    ]


# ── Routes ───────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "ready": rt.embeddings is not None}


@app.post("/api/ask")
async def ask(req: AskRequest):
    """RAG over the pre-indexed demo lecture."""
    try:
        hits   = rt.qdrant_retrieve(req.question)
        answer = rt.rag_inner.invoke({"context": format_context(hits), "question": req.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return {"answer": answer, "sources": _serialize_hits(hits)}


@app.post("/api/upload")
async def upload(audio: UploadFile = File(...)):
    """Transcribe → chunk → embed → store in a per-session in-memory index."""
    suffix = Path(audio.filename or "").suffix or ".audio"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(await audio.read())
        path = f.name

    try:
        result = transcribe(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try: os.unlink(path)
        except OSError: pass

    duration = float(result["duration"])
    if duration > MAX_UPLOAD_SECONDS + 1:
        raise HTTPException(
            status_code=400,
            detail=f"Audio is {duration:.0f}s — max allowed is {MAX_UPLOAD_SECONDS}s.",
        )

    segments   = result["segments"]
    transcript = " ".join(s["text"] for s in segments).strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="No speech detected.")

    chunks = chunk_segments(segments) or [{"text": transcript, "abs_start": 0.0, "abs_end": duration}]
    docs   = [
        Document(
            page_content=c["text"],
            metadata={"abs_start": c["abs_start"], "abs_end": c["abs_end"], "text": c["text"]},
        )
        for c in chunks
    ]
    store = SimpleVectorStore(rt.embeddings)
    store.add_documents(docs)
    session_id = uuid.uuid4().hex
    rt.sessions[session_id] = store

    try:
        summary = rt.summary_chain.invoke({"text": transcript}).strip()
    except Exception as e:
        summary = f"⚠️ Couldn't summarize: {e}"

    return {
        "session_id": session_id,
        "transcript": transcript,
        "summary":    summary,
        "duration":   duration,
        "device":     result.get("device", "unknown"),
        "num_chunks": len(chunks),
    }


@app.post("/api/ask-upload")
async def ask_upload(req: AskUploadRequest):
    store = rt.sessions.get(req.session_id)
    if store is None:
        raise HTTPException(
            status_code=404,
            detail="Session expired or not found. Upload a recording first.",
        )
    hits = store.similarity_search_with_score(req.question, k=3)
    try:
        answer = rt.rag_inner.invoke({"context": format_context(hits), "question": req.question})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    return {"answer": answer, "sources": _serialize_hits(hits)}
