---
language:
  - ar
  - en
license: mit
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
  - whisper
  - asr
  - code-switching
  - arabic
  - egyptian-arabic
  - speech-recognition
base_model: openai/whisper-medium
datasets:
  - Seif-Eldeen-Sameh/asr_codeswitched_dataset
metrics:
  - wer
---

<!-- Upload this file as the README.md of
     https://huggingface.co/Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched -->

# Whisper-medium · Arabic/English Code-Switched ASR

Fine-tuned [`openai/whisper-medium`](https://huggingface.co/openai/whisper-medium) for
**code-switched Egyptian Arabic + English** speech — the register used in Arab-world
university lectures, where dialectal Arabic and English technical terms alternate mid-sentence.

A **CTranslate2 INT8** build for fast CPU/edge inference is available at
[`whisper-medium-arabic-codeswitched-ct2`](https://huggingface.co/Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched-ct2).

## Why fine-tune

Base Whisper tends to transliterate or drop English technical terms in Arabic speech
("embedding" → "إمبدنغ" or omitted) and pushes dialect toward Modern Standard Arabic.
For downstream retrieval over lecture content, the original surface form matters — so the
model is trained to keep code-switching intact.

## Intended use

- Transcribing code-switched Arabic/English lectures, talks, and meetings.
- Upstream component of an ASR → RAG pipeline (see
  [Muhadara RAG](https://huggingface.co/spaces/Seif-Eldeen-Sameh/muhadara-rag)).

Not intended for: high-stakes, legal, or medical transcription without human review.

## Training

| | |
|---|---|
| Base model | `openai/whisper-medium` (769M) |
| Data | [`asr_codeswitched_dataset`](https://huggingface.co/datasets/Seif-Eldeen-Sameh/asr_codeswitched_dataset) (~65–80 h) |
| Steps | 6,000 |
| Batch | 2 × grad-accum 8 (effective 16) |
| LR | 1e-5, 500 warmup steps |
| Precision | fp16 |
| Selection | best WER on held-out eval |

## Usage

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

proc  = WhisperProcessor.from_pretrained("Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched")
model = WhisperForConditionalGeneration.from_pretrained("Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched")

audio, sr = librosa.load("clip.wav", sr=16000, mono=True)
feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
ids   = model.generate(feats, language="ar", task="transcribe")
print(proc.batch_decode(ids, skip_special_tokens=True)[0])
```

Fast CPU inference with faster-whisper (CT2 build):

```python
from faster_whisper import WhisperModel
m = WhisperModel("Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched-ct2",
                 device="cpu", compute_type="int8")
segs, _ = m.transcribe("clip.wav", language="ar", beam_size=1, vad_filter=True)
print(" ".join(s.text for s in segs))
```

## Evaluation

Measured on a held-out slice of [`asr_codeswitched_dataset`](https://huggingface.co/datasets/Seif-Eldeen-Sameh/asr_codeswitched_dataset) with `jiwer` WER, light text normalization (lowercase, diacritic + punctuation strip, no MSA folding). Reproduce with [`eval/evaluation.ipynb`](https://github.com/Seif-Eldeen-Sameh/muhadara-rag/blob/main/eval/evaluation.ipynb).

| Model | WER ↓ |
|---|---|
| `openai/whisper-medium` (base) | 52.4 % |
| `whisper-medium-arabic-codeswitched` (this) | **17.9 %** |
| Absolute reduction | **−34.5 points** |

### Latency (40.8 s clip, median of 3 runs)

| Variant | Device | Wall time | RTF |
|---|---|---|---|
| CT2 INT8 | CPU (2 vCPU) | 82.94 s | 2.03× |
| CT2 INT8 | T4 GPU | 2.49 s | 0.061× |
| CT2 FP16 | T4 GPU | 2.34 s | 0.057× |

The CT2 INT8 build is ~4× smaller than the original FP32 (~380 MB vs ~3 GB) and ~33× faster on T4 GPU vs free-tier CPU.

## Limitations

- Tuned for Egyptian dialect; other dialects may degrade.
- Inherits Whisper's tendency to hallucinate on long silences (mitigated downstream with VAD +
  repetition guards).
- Far-field / very noisy recordings remain hard (see the Muhadara RAG repo's notes on audio
  enhancement experiments).

## Citation

```bibtex
@misc{muhadara_whisper_2026,
  author = {Seif Eldeen Sameh},
  title  = {Whisper-medium Arabic/English Code-Switched ASR},
  year   = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched}}
}
```
