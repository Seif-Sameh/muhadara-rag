# Architecture & Design Notes

Detailed companion to the [README](../README.md). This documents *why* the system is shaped the way it is. The high-level figure lives at [`assets/architecture.png`](../assets/architecture.png).

## Runtime overview

Two parallel pipelines run inside the deployed app, both originating from the same user:

1. **Upload pipeline** — turns an uploaded audio clip into searchable indexed chunks (Modal GPU ASR → chunk → embed → vector store).
2. **Query pipeline** — turns a text question into a grounded answer with timestamp citations (embed query → cosine search → Groq RAG).

The two pipelines meet at the vector store: whatever was indexed by the upload pipeline becomes context for the next query.

## Component notes

### Ingestion & chunking
- All input normalized to 16 kHz mono via ffmpeg (`librosa`/`faster-whisper` handle this transparently).
- Long pre-indexed lectures use 30 s chunks with **5 s overlap** so no word is severed at a boundary; short user uploads (≤ 2 min) are processed in one pass on Modal.
- Each chunk carries absolute `(start_sec, end_sec)` timestamps that travel the whole pipeline — this is what makes timestamp-grounded citations possible.

### ASR
- Fine-tuned `whisper-medium-arabic-codeswitched` → **CTranslate2 INT8** (~380 MB, ~4× smaller than FP32).
- WER measured on a held-out code-switched test slice: **17.9 %** (vs. 52.4 % base whisper-medium — a 34.5 point reduction). See [`eval/evaluation.ipynb`](../eval/evaluation.ipynb).
- RMS-based VAD skips silent chunks; a 3-gram repetition guard catches Whisper hallucination loops.
- Word-level timestamps are offset by the chunk start so every word resolves to an absolute source-file position.

### Deduplication (pre-baked indexing only)
The 5 s overlap means consecutive chunks share text at the boundary. `deduplicate_segments` normalizes both sides, finds the longest matching prefix between the previous chunk's tail and the current chunk's head (Levenshtein-gated), and strips it — while keeping the surviving text's timestamps intact. User uploads (≤ 2 min, one ASR call) don't need this step.

### Semantic re-chunking
ASR segments are uneven (sentences, half-sentences). They're merged/split into 30–120 word blocks so embeddings capture a coherent unit of meaning, with the block's `(abs_start, abs_end)` derived from its constituent segments.

### LLM correction
Constrained Groq prompt: fix homophones / splits / punctuation, **never** MSA-ify dialect or translate English terms. A length-ratio guard (`0.5 ≤ corrected/original ≤ 2.0`) rejects runaway rewrites and falls back to the original text. Skipped on the live upload path to keep response latency low; applied during pre-baked indexing where it has a few minutes to run.

### Embedding
`intfloat/multilingual-e5-large`, 1024-d. e5 expects a `passage:` prefix for documents and `query:` prefix for searches — we wrap that inside the `E5Embeddings` class so the rest of the code doesn't have to think about it.

### Vector store
- **Pre-baked content** lives in Qdrant Cloud (free tier).
- **User uploads** live in a per-session in-memory store (`SimpleVectorStore` — 20 lines of numpy cosine similarity). No Qdrant pollution from random visitors; sessions die with the browser tab.

### Retrieval
Same retrieval function regardless of source: cosine top-k, payload tolerant of both flat (older Qdrant upserts) and nested-`metadata` (LangChain default) layouts so timestamps are always findable.

### LLM (RAG answer)
Groq `gpt-oss-120b` via `langchain-groq`. Prompt forces the model to (a) use only the provided context, (b) cite the source timestamp in `[MM:SS]` verbatim for every claim, (c) respond in the same language as the question, (d) say "not in context" rather than hallucinate.

## Serving topology

| Concern | Where it runs | Rationale |
|---|---|---|
| UI, retrieval, LLM API calls | HF Space (free CPU) | Cheap, always on; these are light. |
| GPU transcription | Modal (T4, scale-to-zero) | Expensive but bursty — pay only on upload, free monthly credit covers a demo. |
| Vector search | Qdrant Cloud | Managed, stateful, free tier. |
| LLM | Groq API | Fast, free tier. |

The frontend calls Modal over HTTP. If `MODAL_ASR_URL` is unset or the call fails, it transparently falls back to local CPU `faster-whisper` — no hard dependency on the GPU service.

## Deliberate non-goals (v1)
- No user accounts or upload persistence (sessions are in-memory).
- No multi-lecture corpus surface; the global Qdrant collection exists as a fallback retrieval target but isn't a user-visible feature.
- No real-time streaming ASR (batch transcription of finite clips only).
- LLM correction skipped on upload path (latency tradeoff).
