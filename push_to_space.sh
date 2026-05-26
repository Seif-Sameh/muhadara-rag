#!/usr/bin/env bash
# One-shot push of this folder to the Hugging Face Space.
# Usage:
#   1. Run `huggingface-cli login` first (or set HF_TOKEN env var) — needs a WRITE token.
#   2. From inside the muhadara-rag/ folder: ./push_to_space.sh "first commit message"
#
# Notes:
# - You only need to set the secrets (QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY) ONCE
#   via the HF Space UI: https://huggingface.co/spaces/Seif-Eldeen-Sameh/muhadara-rag/settings
# - If assets/demo.mp3 is large (>10MB), enable Git LFS for it (see below).

set -euo pipefail

SPACE_REPO="https://huggingface.co/spaces/Seif-Eldeen-Sameh/muhadara-rag"
MSG="${1:-Initial app}"

if [ ! -d ".git" ]; then
    echo "▶ Initializing git repo and adding HF remote…"
    git init -b main
    git remote add origin "$SPACE_REPO"

    # Pull the existing README from the Space so we don't conflict with the one HF created
    git fetch origin main
    git reset --soft origin/main || true
fi

# Optional: track large audio files with Git LFS
if [ -f "assets/demo.mp3" ]; then
    size_mb=$(du -m "assets/demo.mp3" | cut -f1)
    if [ "$size_mb" -gt 10 ]; then
        echo "▶ demo.mp3 is ${size_mb}MB — tracking with Git LFS"
        git lfs install
        git lfs track "assets/*.mp3" "assets/*.wav" "assets/*.m4a"
        git add .gitattributes
    fi
fi

git add .
git commit -m "$MSG" || echo "(nothing to commit)"
echo "▶ Pushing to $SPACE_REPO …"
git push -u origin main
echo "✅ Done. Build status: https://huggingface.co/spaces/Seif-Eldeen-Sameh/muhadara-rag"
