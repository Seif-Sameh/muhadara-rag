"""Muhadara RAG — HF Spaces app (Gradio Blocks).

Tabs:
  1. Demo Lecture     — RAG over a pre-indexed lecture (Qdrant + Groq, CPU, instant)
  2. Upload Your Own  — ASR on Modal GPU (CPU fallback) → in-memory RAG over your clip
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from langchain_core.documents import Document

from utils import (
    E5Embeddings,
    SimpleVectorStore,
    MAX_UPLOAD_SECONDS,
    build_chains,
    chunk_segments,
    format_context,
    format_timestamp,
    load_vector_stores,
    make_qdrant_retriever,
    transcribe,
)

# ── gradio_client schema-walker hardening ────────────────────
# LangChain runnables emit JSON schemas with `additionalProperties: True` (bool),
# which Gradio's /info walker can't handle. Patch the leaf functions defensively.
import gradio_client.utils as _gcu
_orig_internal, _orig_gettype = _gcu._json_schema_to_python_type, _gcu.get_type

def _safe_internal(s, d=None):
    if isinstance(s, bool): return "Any"
    try:    return _orig_internal(s, d)
    except Exception: return "Any"

def _safe_gettype(s):
    if isinstance(s, bool): return "Any"
    try:    return _orig_gettype(s)
    except Exception: return "Any"

_gcu._json_schema_to_python_type = _safe_internal
_gcu.get_type = _safe_gettype

# ── Config (Space secrets) ───────────────────────────────────
QDRANT_URL     = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "").strip()

DEMO_LECTURE_STEM  = "lec1"
DEMO_LECTURE_TITLE = "NLP Lecture 1 — Syntax, Semantics & Ambiguity"
DEMO_AUDIO_PATH    = "assets/demo.mp3"

_missing = [k for k, v in {
    "QDRANT_URL": QDRANT_URL, "QDRANT_API_KEY": QDRANT_API_KEY, "GROQ_API_KEY": GROQ_API_KEY,
}.items() if not v]
if _missing:
    raise RuntimeError(f"Missing secret(s): {_missing}. Set them in the Space settings.")

# ── Cold-start init ──────────────────────────────────────────
print("⏳ Loading embeddings …")
embeddings = E5Embeddings()

print("⏳ Connecting Qdrant …")
per_lecture_store, global_store = load_vector_stores(
    embeddings, DEMO_LECTURE_STEM, QDRANT_URL, QDRANT_API_KEY,
)

print("⏳ Building chains …")
rag_inner, summary_chain = build_chains(GROQ_API_KEY)
qdrant_retrieve = make_qdrant_retriever(per_lecture_store, global_store, embeddings)

# Local CPU ASR is loaded lazily only if Modal is unavailable (see utils.transcribe)
_local_asr = None
print("✅ App ready")


# ── Demo tab ─────────────────────────────────────────────────
def on_ask(question, history):
    question = (question or "").strip()
    if not question:
        return history, "", ""
    try:
        hits   = qdrant_retrieve(question)
        answer = rag_inner.invoke({"context": format_context(hits), "question": question})
    except Exception as e:
        return (history or []) + [(question, f"⚠️ Error: {e}")], "", ""
    history = (history or []) + [(question, answer)]
    chunks_md = "### Retrieved chunks\n\n"
    for doc, score in hits:
        ts = format_timestamp(float(doc.metadata.get("abs_start", 0)))
        body = doc.page_content[:280] + ("…" if len(doc.page_content) > 280 else "")
        chunks_md += f"- **`[{ts}]`** — score {score:.3f}  \n  {body}\n\n"
    return history, "", chunks_md


# ── Upload tab ───────────────────────────────────────────────
def index_upload(audio_path):
    global _local_asr
    if not audio_path:
        return "", "", "⚠️ Please upload an audio clip first.", None

    try:
        result = transcribe(audio_path, local_asr=_local_asr)
    except Exception as e:
        return "", "", f"⚠️ Transcription failed: {e}", None

    segments   = result["segments"]
    duration   = result["duration"]
    device     = result["device"]
    transcript = " ".join(s["text"] for s in segments).strip()

    if duration > MAX_UPLOAD_SECONDS + 1:
        return "", "", f"⚠️ Clip is {duration:.0f}s — max allowed is {MAX_UPLOAD_SECONDS}s.", None
    if not transcript:
        return "*(no speech detected)*", "", "⚠️ No speech detected.", None

    chunks = chunk_segments(segments) or [
        {"text": transcript, "abs_start": 0.0, "abs_end": duration}
    ]
    docs = [Document(page_content=c["text"],
                     metadata={"abs_start": c["abs_start"], "abs_end": c["abs_end"], "text": c["text"]})
            for c in chunks]
    store = SimpleVectorStore(embeddings)
    store.add_documents(docs)

    try:
        summary = summary_chain.invoke({"text": transcript}).strip()
    except Exception as e:
        summary = f"⚠️ Couldn't summarize: {e}"

    status = (f"✅ Transcribed {duration:.1f}s on **{device}** · indexed {len(chunks)} chunk"
              f"{'s' if len(chunks) != 1 else ''}. Ask anything below.")
    return transcript, summary, status, store


def ask_upload(question, history, store):
    question = (question or "").strip()
    if not question:
        return history, ""
    if store is None:
        return (history or []) + [(question, "⚠️ Upload and transcribe a recording first.")], ""
    hits = store.similarity_search_with_score(question, k=3)
    try:
        answer = rag_inner.invoke({"context": format_context(hits), "question": question})
    except Exception as e:
        answer = f"⚠️ Error: {e}"
    return (history or []) + [(question, answer)], ""


# ── UI ───────────────────────────────────────────────────────
CSS = Path("style.css").read_text() if Path("style.css").exists() else ""
theme = gr.themes.Soft(
    primary_hue="emerald", secondary_hue="sky", neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

with gr.Blocks(css=CSS, theme=theme, title="Muhadara RAG") as app:
    gr.HTML("""
    <div class="app-header">
        <h1>🎙️ Muhadara RAG</h1>
        <p class="tagline">
            Search Arabic/English code-switched university lectures by meaning.<br/>
            Ask a question — get answers with timestamps pointing back to the audio.
        </p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("📚 Demo Lecture"):
            gr.Markdown(
                f"### {DEMO_LECTURE_TITLE}\n"
                "Pre-indexed. Ask anything in **Arabic or English** — answers cite timestamps."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    if Path(DEMO_AUDIO_PATH).exists():
                        gr.Audio(value=DEMO_AUDIO_PATH, label="Demo lecture audio", interactive=False)
                    chatbot = gr.Chatbot(height=420, show_label=False)
                    with gr.Row():
                        question = gr.Textbox(placeholder="Ask in Arabic or English…",
                                              show_label=False, scale=5, container=False)
                        ask_btn = gr.Button("Ask", variant="primary", scale=1)
                with gr.Column(scale=1):
                    chunks_panel = gr.Markdown("*Ask a question to see retrieved chunks here.*")

            gr.Examples(
                examples=[
                    "What is NLP and why is it difficult?",
                    "ما الفرق بين ال syntax و ال semantics؟",
                    "What is ambiguity in natural language?",
                    "اشرح ال parsing tree",
                    "What is wordnet?",
                ],
                inputs=question, label="Try one",
            )
            ask_btn.click(on_ask, [question, chatbot], [chatbot, question, chunks_panel])
            question.submit(on_ask, [question, chatbot], [chatbot, question, chunks_panel])

        with gr.Tab("🎤 Upload Your Own"):
            upload_store_state = gr.State(value=None)
            gr.Markdown(
                f"### Try it on your own audio\n"
                f"Upload an Arabic/English clip (max **{MAX_UPLOAD_SECONDS}s**). "
                f"It's transcribed on a GPU, summarized, and indexed — then ask questions about it."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                           label="Your audio clip")
                    transcribe_btn = gr.Button("Transcribe + Index", variant="primary")
                    status_md = gr.Markdown()
                    transcript_out = gr.Textbox(label="Transcript", lines=8, max_lines=18, show_copy_button=True)
                    summary_out    = gr.Textbox(label="Summary", lines=4, max_lines=10, show_copy_button=True)
                with gr.Column(scale=1):
                    gr.Markdown("#### 💬 Ask about this recording")
                    upload_chatbot = gr.Chatbot(height=420, show_label=False)
                    with gr.Row():
                        upload_question = gr.Textbox(placeholder="What did the speaker say about…?",
                                                     show_label=False, scale=5, container=False)
                        upload_ask_btn = gr.Button("Ask", variant="primary", scale=1)

            transcribe_btn.click(index_upload, [audio_input],
                                 [transcript_out, summary_out, status_md, upload_store_state])
            upload_ask_btn.click(ask_upload, [upload_question, upload_chatbot, upload_store_state],
                                 [upload_chatbot, upload_question])
            upload_question.submit(ask_upload, [upload_question, upload_chatbot, upload_store_state],
                                   [upload_chatbot, upload_question])

    gr.HTML("""
    <div class="app-footer">
        Fine-tuned Whisper-medium (CT2 INT8) · multilingual-e5-large · Qdrant · Groq · Modal GPU · LangChain<br/>
        Built by <a href="https://huggingface.co/Seif-Eldeen-Sameh">@Seif-Eldeen-Sameh</a>
    </div>
    """)


if __name__ == "__main__":
    app.launch(show_api=False)
