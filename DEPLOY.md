# Deployment Runbook

Do these in order. Steps A–E get you a live, GPU-accelerated, CI/CD-deployed app.
F–H are polish (demo audio, cards, metrics).

Prereqs: a Hugging Face **write** token, a GitHub account, a Modal account (free), and the
`gh` CLI (`brew install gh`).

---

## A. Publish the CT2 model to the HF Hub

The Space and Modal both pull the quantized model from the Hub (not Drive). Run in Colab:

```python
!pip install -q huggingface_hub
from huggingface_hub import HfApi, login
login(token="hf_xxxYOUR_WRITE_TOKENxxx")

repo = "Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched-ct2"
HfApi().create_repo(repo, repo_type="model", exist_ok=True)
HfApi().upload_folder(
    folder_path="/content/drive/MyDrive/whisper_ct2_model",
    repo_id=repo, repo_type="model",
    commit_message="CT2 INT8 quantized fine-tuned Whisper",
)
print("✅ pushed")
```

## B. Deploy the Modal GPU ASR service

```bash
pip install modal
modal token new                              
modal secret create muhadara-asr-token ASR_TOKEN=$(openssl rand -hex 16)
modal deploy modal_asr.py
```

`modal deploy` prints a URL like
`https://seif-eldeen-sameh--muhadara-asr-whisper-transcribe.modal.run`.
Copy it — that's `MODAL_ASR_URL`. The `ASR_TOKEN` you generated is `MODAL_ASR_TOKEN`
(view it later with `modal secret list` / the Modal dashboard).
   
## C. Create the GitHub repo

```bash
cd muhadara-rag
git init -b main
git add .
git commit -m "Muhadara RAG: end-to-end code-switched lecture RAG"
gh repo create Seif-Eldeen-Sameh/muhadara-rag --public --source=. --push
```

Add the HF token so CI/CD can deploy:

```bash
gh secret set HF_TOKEN          # paste your HF write token
```

## D. Set the HF Space secrets

Space → **Settings → Variables and secrets** → add (as *secrets*):

| Key | Value |
|---|---|
| `QDRANT_URL` | your Qdrant Cloud URL |
| `QDRANT_API_KEY` | your Qdrant key |
| `GROQ_API_KEY` | your Groq key |
| `HF_CT2_MODEL` | `Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched-ct2` |
| `MODAL_ASR_URL` | the URL from step B |
| `MODAL_ASR_TOKEN` | the token from step B |

(If you skip B, leave the two `MODAL_*` blank — the app falls back to CPU transcription.)

## E. Deploy to the Space

The GitHub Action (`.github/workflows/deploy.yml`) auto-pushes to the Space on every push to
`main`. Trigger the first run:

```bash
git commit --allow-empty -m "trigger deploy" && git push
```

Watch it under the repo's **Actions** tab. When green, the Space rebuilds and goes live at
https://huggingface.co/spaces/Seif-Eldeen-Sameh/muhadara-rag

---

## F. Add the demo audio (for the audio player)

```bash
# Convert the lecture to a compact mono MP3 (~37 MB for 82 min @ 64 kbps)
ffmpeg -i "../Nlp le c1.m4a" -ac 1 -ar 16000 -b:a 64k assets/demo.mp3

git lfs install
git lfs track "assets/*.mp3"
git add .gitattributes assets/demo.mp3
git commit -m "Add demo lecture audio (LFS)"
git push
```

## G. Upload the model & dataset cards

```bash
# Model card
huggingface-cli upload Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched \
    cards/MODEL_CARD.md README.md --repo-type model
# Dataset card
huggingface-cli upload Seif-Eldeen-Sameh/asr_codeswitched_dataset \
    cards/DATASET_CARD.md README.md --repo-type dataset
```

## H. Fill in the metrics

Run `eval/evaluation.ipynb` (Colab GPU runtime). Copy the WER + latency numbers into the
`TODO`s in `README.md` and `cards/MODEL_CARD.md`, then commit.

---

## Done — what to share on LinkedIn

- The **live Space link** (works for anyone, no signup)
- The **GitHub repo** (README is the portfolio piece)
- The **architecture diagram** (renders on GitHub from the README's Mermaid block)
- One line that lands: *"End-to-end code-switched ASR + RAG: fine-tuned & quantized Whisper,
  serverless GPU inference (scale-to-zero), vector search with timestamp-grounded answers,
  CI/CD to a live demo."*
