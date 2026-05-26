"""Serverless GPU ASR microservice for Muhadara RAG (Modal).

Why this exists
---------------
The HF Space frontend runs on free CPU — fine for retrieval + LLM calls, but slow
for Whisper. This offloads transcription to an on-demand GPU that **scales to zero**:
a container only spins up when someone actually uploads audio, then shuts down after
5 minutes idle. Modal's free monthly credit ($30) covers a demo comfortably.

Deploy
------
    pip install modal
    modal token new                      # one-time auth
    modal deploy modal_asr.py

That prints a public URL, e.g.
    https://<your-user>--muhadara-asr-whisper-transcribe.modal.run

Put it (and a shared token) in the HF Space secrets:
    MODAL_ASR_URL   = <that URL>
    MODAL_ASR_TOKEN = <any random string; also set here as ASR_TOKEN secret>

Architecture
------------
    HF Space (CPU, Gradio)  --POST audio bytes-->  Modal (T4 GPU, faster-whisper)
                            <--JSON segments-----
"""
import os
import modal

APP_NAME = "muhadara-asr"
MODEL_ID = "Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched-ct2"

app = modal.App(APP_NAME)

# Image: ffmpeg for decoding + faster-whisper (CTranslate2 backend)
# CUDA + cuDNN base image — faster-whisper / CTranslate2 need libcublas + libcudnn
# at runtime when device="cuda". `debian_slim` ships without them.
# NVIDIA changed tag scheme: it's `-cudnn-runtime` now, not `-cudnn9-runtime`.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper==1.0.3",
        "huggingface_hub>=0.23",
        "fastapi[standard]",
        "requests>=2.31",   # faster-whisper imports it at module load time
    )
)

# Make fastapi types resolvable inside the container only — so `modal deploy`
# from your laptop doesn't require fastapi locally.
with image.imports():
    from fastapi import Request, Response

# Persist the downloaded CT2 model across cold starts (no re-download each spin-up)
model_cache = modal.Volume.from_name("muhadara-whisper-cache", create_if_missing=True)
CACHE_DIR = "/cache"


@app.cls(
    image=image,
    gpu="T4",                       # cheapest GPU; plenty for whisper-medium INT8
    scaledown_window=300,           # stay warm 5 min after last call, then scale to zero
    volumes={CACHE_DIR: model_cache},
    # No auth secret — endpoint is open. Acceptable for a portfolio demo because
    # Modal's free monthly credit caps spend. Add the secret back if you ever
    # share the URL widely.
)
class Whisper:
    @modal.enter()
    def load(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            MODEL_ID,
            device="cuda",
            compute_type="int8_float16",   # INT8 weights, FP16 compute — fast on T4
            download_root=CACHE_DIR,
        )
        print("✅ Whisper CT2 model loaded on GPU")

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def transcribe(self, request: Request):
        """POST raw audio bytes (any ffmpeg-decodable format). Returns JSON segments."""
        import json, tempfile, os as _os

        audio_bytes = await request.body()
        if not audio_bytes:
            return Response(content=json.dumps({"error": "empty body"}),
                            status_code=400, media_type="application/json")

        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
            f.write(audio_bytes)
            path = f.name
        try:
            segs, info = self.model.transcribe(
                path, language="ar", beam_size=1, vad_filter=True,
            )
            segments = [
                {"text": s.text.strip(), "start": float(s.start), "end": float(s.end)}
                for s in segs
            ]
            return {
                "segments": segments,
                "duration": float(info.duration),
                "device": "modal-t4-gpu",
            }
        finally:
            _os.unlink(path)


@app.local_entrypoint()
def smoke_test():
    """`modal run modal_asr.py` — quick check the container builds and loads."""
    print(f"App '{APP_NAME}' OK. Deploy with: modal deploy modal_asr.py")
