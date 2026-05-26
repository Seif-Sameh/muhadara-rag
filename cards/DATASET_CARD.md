---
language:
  - ar
  - en
license: mit
task_categories:
  - automatic-speech-recognition
tags:
  - code-switching
  - arabic
  - egyptian-arabic
  - speech
pretty_name: Arabic/English Code-Switched ASR Dataset
size_categories:
  - 10K<n<100K
---

<!-- Upload this as the README.md of
     https://huggingface.co/datasets/Seif-Eldeen-Sameh/asr_codeswitched_dataset -->

# Arabic/English Code-Switched ASR Dataset

Audio + transcripts of **code-switched Egyptian Arabic and English** speech, assembled to
fine-tune ASR for Arab-world lecture content where dialectal Arabic and English technical
vocabulary alternate within sentences.

## Composition

| Source | Description |
|---|---|
| EJUST custom recordings | Locally recorded/segmented code-switched clips (`segments_codeswitched.csv` + WAVs) |
| [`MohamedRashad/arabic-english-code-switching`](https://huggingface.co/datasets/MohamedRashad/arabic-english-code-switching) | ~12,480 public code-switched clips (≤25 s each) |

Total: **~65–80 hours** of audio (run the snippet below for the exact figure).

## Schema

| Column | Type | Description |
|---|---|---|
| `audio` | `Audio(16kHz)` | Mono waveform |
| `transcript` | `string` | Code-switched transcript (dialect + English terms preserved) |

## Load

```python
from datasets import load_dataset, Audio
ds = load_dataset("Seif-Eldeen-Sameh/asr_codeswitched_dataset", split="train")
ds = ds.cast_column("audio", Audio(sampling_rate=16000))
```

### Exact duration

```python
total = sum(len(r["audio"]["array"]) / r["audio"]["sampling_rate"] for r in ds)
print(f"{total/3600:.1f} hours over {len(ds):,} examples")
```

## Intended use

Fine-tuning ASR (e.g. Whisper) for code-switched Arabic/English. See the model trained on it:
[`whisper-medium-arabic-codeswitched`](https://huggingface.co/Seif-Eldeen-Sameh/whisper-medium-arabic-codeswitched).

## Limitations & ethics

- Egyptian dialect dominant; not representative of all Arabic dialects.
- Lecture/talk domain; technical vocabulary skew.
- Ensure you have rights to any audio you add; respect speaker consent.

## License

MIT (custom portion). The public source dataset retains its own license — review before
redistribution.
