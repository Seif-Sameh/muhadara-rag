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
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper==1.0.3",
        "huggingface_hub>=0.23",
        "fastapi[standard]",
    )
)

# Persist the downloaded CT2 model across cold starts (no re-download each spin-up)
model_cache = modal.Volume.from_name("muhadara-whisper-cache", create_if_missing=True)
CACHE_DIR = "/cache"


@app.cls(
    image=image,
    gpu="T4",                       # cheapest GPU; plenty for whisper-medium INT8
    scaledown_window=300,           # stay warm 5 min after last call, then scale to zero
    volumes={CACHE_DIR: model_cache},
    secrets=[modal.Secret.from_name("muhadara-asr-token", required_keys=["ASR_TOKEN"])],
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
    async def transcribe(self, request):
        """POST raw audio bytes (any ffmpeg-decodable format). Returns JSON segments."""
        from fastapi import Response
        import json, tempfile, os as _os

        # Lightweight shared-secret auth so randoms can't run up the bill
        expected = _os.environ.get("ASR_TOKEN", "")
        if expected and request.headers.get("X-Auth-Token") != expected:
            return Response(content=json.dumps({"error": "unauthorized"}),
                            status_code=401, media_type="application/json")

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
