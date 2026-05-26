"""Pipeline helpers for the Muhadara RAG Space.

Keeps app.py lean. Heavy lifting (model loading, vector stores, chains, the
Modal ASR client) lives here.

Inference layout:
  - Demo lecture tab  → retrieval over Qdrant + Groq LLM   (CPU, instant)
  - Upload tab        → ASR on Modal GPU (CPU fallback) + in-memory RAG
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import requests

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq


# ── Constants ────────────────────────────────────────────────
MAX_UPLOAD_SECONDS = 120
GROQ_MODEL         = "openai/gpt-oss-120b"
EMBED_MODEL        = "intfloat/multilingual-e5-large"
EMBED_DIM          = 1024
TOP_K              = 5
FALLBACK_THRESHOLD = 0.70
GLOBAL_COLLECTION  = "lectures_global"

# Modal serverless GPU endpoint (set in Space secrets). Empty → CPU fallback.
MODAL_ASR_URL   = os.environ.get("MODAL_ASR_URL", "").strip()
MODAL_ASR_TOKEN = os.environ.get("MODAL_ASR_TOKEN", "").strip()


# ── Formatting ───────────────────────────────────────────────
def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem  = divmod(seconds, 3600)
    m, s    = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── Embeddings ───────────────────────────────────────────────
class E5Embeddings(Embeddings):
    """e5 needs 'passage: ' for docs and 'query: ' for queries."""

    def __init__(self, model_name: str = EMBED_MODEL):
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=16, normalize_embeddings=True, show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            f"query: {text}",
            normalize_embeddings=True, show_progress_bar=False,
        ).tolist()


# ── Qdrant (demo lecture) ────────────────────────────────────
def load_vector_stores(embeddings: Embeddings, demo_stem: str,
                       qdrant_url: str, qdrant_api_key: str):
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    per_lecture = QdrantVectorStore(
        client=client, collection_name=f"lecture_{demo_stem}",
        embedding=embeddings, content_payload_key="text",
    )
    globl = QdrantVectorStore(
        client=client, collection_name=GLOBAL_COLLECTION,
        embedding=embeddings, content_payload_key="text",
    )
    per_lecture._raw_client = client
    globl._raw_client = client
    return per_lecture, globl


# ── In-memory store (uploaded clips) ─────────────────────────
class SimpleVectorStore:
    """Cosine-similarity in-memory store for ephemeral uploaded recordings.

    We don't write user uploads to Qdrant — that would pollute the cloud DB with
    one collection per random visitor. This lives only for the session.
    """
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings
        self.entries: list[tuple[np.ndarray, Document]] = []

    def add_documents(self, documents: list[Document]):
        if not documents:
            return
        vecs = self.embeddings.embed_documents([d.page_content for d in documents])
        for v, d in zip(vecs, documents):
            self.entries.append((np.asarray(v, dtype=np.float32), d))

    def similarity_search_with_score(self, query: str, k: int = 3):
        if not self.entries:
            return []
        qv = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        scored = [(d, float(np.dot(qv, v))) for v, d in self.entries]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]


# ── ASR: Modal GPU with CPU fallback ─────────────────────────
def load_local_asr():
    """Lazily load a local CPU faster-whisper as the fallback path."""
    from faster_whisper import WhisperModel
    model_id = os.environ.get(
        "HF_CT2_MODEL", "Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched-ct2"
    )
    return WhisperModel(model_id, device="cpu", compute_type="int8",
                        cpu_threads=os.cpu_count() or 4)


def transcribe(audio_path: str, local_asr=None) -> dict[str, Any]:
    """Transcribe via Modal GPU if configured, else local CPU. Returns segments + meta."""
    # ── Primary: Modal serverless GPU ──
    if MODAL_ASR_URL:
        try:
            with open(audio_path, "rb") as f:
                resp = requests.post(
                    MODAL_ASR_URL,
                    data=f.read(),
                    headers={"X-Auth-Token": MODAL_ASR_TOKEN} if MODAL_ASR_TOKEN else {},
                    timeout=180,
                )
            resp.raise_for_status()
            data = resp.json()
            return {
                "segments": data["segments"],
                "duration": data.get("duration", 0.0),
                "device":   data.get("device", "modal-gpu"),
            }
        except Exception as e:
            print(f"[ASR] Modal failed ({e}); falling back to local CPU")

    # ── Fallback: local CPU ──
    import librosa
    if local_asr is None:
        local_asr = load_local_asr()
    audio, sr = librosa.load(audio_path, sr=16_000, mono=True)
    duration = len(audio) / sr
    segs, _ = local_asr.transcribe(audio, language="ar", beam_size=1, vad_filter=True)
    segments = [{"text": s.text.strip(), "start": s.start, "end": s.end} for s in segs]
    return {"segments": segments, "duration": duration, "device": "cpu"}


def chunk_segments(segments: list[dict], target_words: int = 40) -> list[dict]:
    """Merge ASR segments into ~target_words chunks with timestamps preserved."""
    chunks, buf, buf_start, wc = [], [], None, 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if buf_start is None:
            buf_start = seg["start"]
        buf.append(text)
        wc += len(text.split())
        if wc >= target_words:
            chunks.append({"text": " ".join(buf), "abs_start": buf_start, "abs_end": seg["end"]})
            buf, buf_start, wc = [], None, 0
    if buf:
        chunks.append({"text": " ".join(buf), "abs_start": buf_start, "abs_end": segments[-1]["end"]})
    return chunks


# ── RAG chain (shared by demo + upload) ──────────────────────
def _payload_to_doc(point) -> tuple[Document, float]:
    payload = dict(point.payload or {})
    nested  = payload.get("metadata")
    if isinstance(nested, dict):
        for k, v in nested.items():
            payload[k] = v
    content = payload.get("text") or payload.get("page_content") or ""
    return Document(page_content=content, metadata=payload), point.score


def build_chains(groq_api_key: str):
    """Returns (rag_inner, summary_chain) — both reusable across tabs."""
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You answer questions about a lecture transcript using ONLY the provided context. "
         "Every factual claim MUST be followed by the source timestamp in MM:SS or HH:MM:SS, "
         "copied verbatim from the context. Do NOT use [00:00] as a placeholder. "
         "Respond in the same language as the question. "
         "If the context does not contain the answer, say so plainly."),
        ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
    ])
    rag_llm   = ChatGroq(model=GROQ_MODEL, api_key=groq_api_key, temperature=0.2, max_tokens=512)
    rag_inner = rag_prompt | rag_llm | StrOutputParser()

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You summarize an Arabic/English code-switched transcript. "
         "Write 2-3 sentences in the SAME language mix. "
         "Preserve technical terms exactly. No preface, no bullets."),
        ("human", "Transcript:\n{text}\n\nSummary:"),
    ])
    summary_llm   = ChatGroq(model=GROQ_MODEL, api_key=groq_api_key, temperature=0.2, max_tokens=256)
    summary_chain = summary_prompt | summary_llm | StrOutputParser()

    return rag_inner, summary_chain


def make_qdrant_retriever(per_lecture_store, global_store, embeddings):
    client = per_lecture_store._raw_client

    def retrieve(query: str):
        q_vec = embeddings.embed_query(query)
        def _search(col):
            resp = client.query_points(col, query=q_vec, limit=TOP_K, with_payload=True)
            return [_payload_to_doc(p) for p in resp.points]
        hits = _search(per_lecture_store.collection_name)
        if hits and hits[0][1] < FALLBACK_THRESHOLD:
            hits = sorted(hits + _search(global_store.collection_name),
                          key=lambda h: -h[1])[:TOP_K]
        return hits

    return retrieve


def format_context(hits) -> str:
    parts = []
    for doc, score in hits:
        ts = format_timestamp(float(doc.metadata.get("abs_start", 0)))
        parts.append(f"[{ts}] (score={score:.3f}) {doc.page_content}")
    return "\n\n".join(parts) if parts else "(no relevant context found)"
