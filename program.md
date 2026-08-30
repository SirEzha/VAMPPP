# Pronunciation Similarity Scoring — Project Roadmap

## 1. Context

Score how closely a **live microphone recording** reproduces the pronunciation of a
**pre-recorded reference line**.

| Constraint | Value |
|---|---|
| Primary language | **Japanese** |
| Stretch languages | Low-resource (e.g. Samoan) — nice-to-have, not required |
| Speakers | Often **non-native**, may not speak the target language |
| Reference audio | Different recording equipment, possibly post-processed |
| Reference text | **Known** (this is decisive — see §2.2) |
| Invariance required | Voice (gender, pitch, timbre) and phrase timing / length / start-end offset |
| Training data | **None** — must be zero-shot, no labeled pronunciation ratings |
| Output | A **single similarity score**, 0–100 |
| Hardware | RTX 3070 Laptop, **8 GB VRAM** |

---

## 2. Design decisions and rationale

### 2.1 Rejected: mel-spectrogram features + global cosine

The original baseline was a magnitude-spectrogram extractor plus cosine similarity,
discarding phase in order to shed voice differences. This inverts the real information
layout:

- **Speaker identity lives in the magnitude spectrum, not phase.** Formant positions
  (vocal-tract length), spectral tilt and F0 harmonic spacing are all magnitude-domain.
  Production speaker-ID systems consume precisely these features — ECAPA-TDNN reaches
  **0.87% EER on VoxCeleb1** from 80-dim log-mel filterbanks. Discarding phase removes
  almost nothing we want gone, and the retained mel spectrogram is close to an optimal
  *speaker fingerprint*: the exact opposite of the goal.
- **A global cosine collapses the signal.** Pronunciation quality is local — *which* phones
  were wrong. One scalar over a pooled embedding mostly reports "same voice, same mic."
- **Channel mismatch hits spectrogram features hardest**, and the reference is explicitly
  channel-mismatched.

**Replacement:** the separating knob is **layer depth inside a self-supervised speech
encoder**, not phase. Layer-wise probing (Pasad et al. 2021) shows phonetic information
peaks around **layer 9 of 12** and collapses after layer 10, while **speaker information
peaks in the early layers**. A CTC **phone posteriorgram (PPG)** is speaker-normalized by
construction and is stronger still.

### 2.2 Decisive: the reference text is known

The primary metric therefore never compares two audio recordings. **Goodness of
Pronunciation (GOP)** scores the live audio against the *intended phone sequence*, so the
reference's microphone and processing are irrelevant to it — **the channel-mismatch problem
largely dissolves.** The reference audio keeps a better job: a **per-phrase calibration
anchor** (§2.5).

### 2.3 Alignment is the mechanism, not an obstacle

CTC forced alignment plus **banded** DTW absorbs different lengths, speaking rates and
start/end offsets with no manual segmentation.

### 2.4 Two Japanese-specific consequences

1. **Duration is contrastive.** Long vowels (`おばさん` vs `おばあさん`) and geminate sokuon
   `っ` are phonemic. Unconstrained DTW warps time away and aligns a short vowel onto a long
   one at near-zero cost — silently erasing a whole class of real L2 errors. **Duration gets
   its own explicit score term**; it is never left to the warper.
2. **Pitch accent is lexical**, so "discard pitch" is not free. Resolution: z-score log-F0
   per utterance — removes gender and absolute pitch, preserves the relative accent contour.
   Kept **opt-in, default off** in v1 because the deliverable is a single number.

### 2.5 Zero-shot calibration via the reference anchor

No labels exist, and phrases differ in intrinsic difficulty and baseline GOP. Fix: run the
**identical pipeline on the reference audio** and report each term *relative to* its
reference value. The native reference defines the per-phrase ceiling, so the score
self-calibrates per line with **no annotation**. Term weights are fixed from the validation
harness (§5), never learned.

### 2.6 Model size

A 2–4B model is the wrong axis: it will not fit in fp16 in 8 GB and buys nothing here.
300M-class SSL encoders are state of the art for frame-level phonetics; 4B speech models are
speech-*LLMs* for captioning/QA, not phone-level detail. Expected peak VRAM **< 2 GB**.

---

## 3. Architecture

```
reference audio ─┐                    ┌─ (anchor) ─┐
                 ├─ preprocess ─ w2v2-xlsr-espeak ─┤            ┌─ GOP term
reference text ──┴─ G2P ─ IPA phones ─┴─ forced align ──────────┼─ duration term ─ combine ─ 0..100
                                                                └─ PPG-DTW term
live audio ──────── preprocess ───────────────────────────────┘
```

**Score terms**

| Term | What it measures | Why it is present |
|---|---|---|
| **GOP** (primary) | Per-phone CTC log-posterior of the target phone over its aligned span, minus logsumexp over all phones | Channel-independent; scores against intended phones |
| **Duration** | Per-mora durations from alignment, normalized by utterance speech rate, live vs reference | Catches Japanese long-vowel / geminate errors DTW cannot see |
| **PPG-DTW** | Length-normalized DTW path cost between live and reference PPGs | Grounds the score in the actual native rendition |
| **F0** *(opt-in, off)* | DTW-aligned correlation of per-utterance z-scored log-F0 | Pitch accent, without gender leakage |

---

## 4. Libraries

| Library | Role | Notes |
|---|---|---|
| `torch`, `torchaudio` **≥2.1** | Core; `functional.forced_align`, `functional.merge_tokens` | Custom CPU/CUDA align kernels, `<star>` token support |
| `transformers` | Loads the acoustic model | fp16 inference |
| **`facebook/wav2vec2-xlsr-53-espeak-cv-ft`** | Acoustic model → IPA CTC posteriors | XLSR-53 multilingual pretrain — correct pick over `lv-60` (English-only) for Japanese and zero-shot transfer |
| `pyopenjtalk` | **Japanese** G2P (`g2p()`, `run_marine=True` for accent) | De facto standard; handles kanji→reading, which eSpeak `ja` does poorly |
| `phonemizer` + `espeak-ng` | G2P for all other languages | Emits IPA matching the model vocabulary |
| `silero-vad` | Leading/trailing silence trim | Not optional — silence frames inflate DTW similarity |
| `pyloudnorm` | Loudness normalize to −23 LUFS | |
| `librosa` / `soundfile` | I/O, resampling, F0 | |
| `speechbrain` (`spkrec-ecapa-voxceleb`) | **Harness only** — speaker embeddings for the invariance probe | Not a runtime dependency |
| `pytest` | Harness runner | |

---

## 5. Roadmap

### M0 — Scaffold
- **Steps:** repo layout (§6), pin dependencies, verify CUDA + fp16 model load on the 3070.
- **Deliverable:** environment reproducible from a lockfile; model loads.
- **Validation:** report peak VRAM at load; assert **< 2 GB**.

### M1 — Preprocess, encoder, G2P
- **Steps:** implement `audio.py` (mono → 16 kHz → −23 LUFS → Silero VAD trim; *identical*
  path for reference and live), `g2p.py` (Japanese via `pyopenjtalk`, others via
  `phonemizer`; static OpenJTalk→eSpeak-IPA mapping table; accept kana/furigana override for
  ambiguous kanji readings; cache per line), `encoder.py` (fp16, emit CTC log-posteriors =
  PPG at 50 Hz, plus layer-9 hidden states behind a flag; per-utterance CMVN then per-frame
  L2 norm on hidden states).
- **Deliverable:** PPG array + IPA phone sequence dumped for one Japanese reference line.
- **Validation:** phone sequence for a known line matches expected kana by inspection;
  PPG argmax decoded to IPA is recognizably the phrase.

### M2 — Forced alignment + GOP
- **Steps:** `align.py` using `forced_align` / `merge_tokens`, run against the same phone
  sequence for **both** live and reference. `score.py` GOP term.
- **Deliverable:** per-phone time spans and per-phone GOP values.
- **Validation:** cut audio at the predicted phone boundaries and confirm by ear; GOP on the
  native reference is high and roughly uniform across phones.

### M3 — Full scorer, single number
- **Steps:** duration term (per-mora, speech-rate normalized), PPG-DTW term (**Sakoe-Chiba
  band ~±30% length ratio**, **symmetric step pattern P=0.5** to block degenerate paths,
  cosine or symmetric-KL local distance), reference-anchored calibration (§2.5), weighted
  combination, fixed monotone squash to 0–100. `api.py`:
  `score(reference_audio, reference_text, live_audio) -> float`.
- **Deliverable:** working end-to-end single score.
- **Validation:** smoke test with one reference line plus a deliberately *good* and
  deliberately *bad* live take — assert good ≫ bad; assert reference-scored-against-itself
  lands at the ceiling.

### M4 — Validation harness ← **the real deliverable**
With zero labels, this is what makes the metric defensible rather than assumed. Probe audio
from TTS voices and/or Common Voice `ja`.

| Probe | Method | Pass criterion |
|---|---|---|
| **Speaker invariance** | Same phrase, N voices/genders; correlate score against ECAPA speaker-embedding distance | scores cluster tight and high; \|r\| ≈ 0 |
| **Channel invariance** | Augment live input: RIR reverb, EQ tilt, band-limiting, MP3/Opus codec, additive noise | score drop below a fixed small threshold |
| **Duration sensitivity** | Shorten long vowels (`ō`→`o`); remove geminate `っ` | score drops materially |
| **Monotonicity** | Inject 1..k phone substitutions (incl. `/r/`→`/l/`) | score decreases monotonically in k |
| **Discrimination** | Score a completely different phrase | score floors |

- **Deliverable:** one probe table, reproducible via `pytest`.
- **Validation / exit gate:** **the build is not "working" until speaker and channel
  invariance pass** — the two requirements the rejected baseline would have silently failed.
  Term weights (§2.5) are fixed from this table's output. A speaker-invariance failure
  indicts the layer choice or normalization; because every term is reported separately, the
  harness localizes which one.

### M5 — Pitch accent (conditional)
- **Trigger:** only if M4 shows accent errors going undetected.
- **Steps:** opt-in F0 term — per-utterance z-scored log-F0, DTW-aligned correlation.
- **Validation:** minimal-pair accent contrasts (e.g. 箸 / 橋) separate; re-run the
  speaker-invariance probe to confirm no gender leakage was reintroduced.

---

## 6. Repo layout

```
src/
  audio.py      # resample, loudness, VAD trim
  g2p.py        # text -> IPA phones (pyopenjtalk / phonemizer)
  encoder.py    # wav2vec2-xlsr-espeak -> PPG + hidden states
  align.py      # CTC forced alignment -> phone spans
  score.py      # GOP / duration / DTW / F0 terms + combination
  api.py        # score() entry point
tests/harness/
  probes.py     # the M4 probe table
  augment.py    # reverb, codec, EQ, noise, phone-swap mutations
data/refs/      # reference audio + text
```

**Output contract:** the headline return is a single 0–100 float, as specified. Per-phone GOP
and per-term subscores are computed anyway — return them on the result object and log them.
Without that breakdown there is no way to distinguish a low score caused by mispronunciation
from one caused by channel mismatch.

---

## 7. Known risks

| Risk | Mitigation |
|---|---|
| Zero-shot IPA quality on Samoan is unmeasurable without data | Treat stretch languages as best-effort; the M4 probes are Japanese-only |
| Kanji reading ambiguity in G2P | Kana/furigana override path in `g2p.py` |
| DTW erases contrastive duration | Explicit duration term + banded warping (§2.4) |
| Pitch accent invisible in v1 | Deliberate, documented v1 exclusion; M5 is the remedy |
| `CLAUDE.md` is auto-loaded every session | Once work is underway, trim to a pointer at a longer `ROADMAP.md` |

---

## References

- [Pasad et al., Layer-wise Analysis of a Self-Supervised Speech Representation Model](https://arxiv.org/abs/2107.04734)
- [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143)
- [facebook/wav2vec2-xlsr-53-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-xlsr-53-espeak-cv-ft)
- [torchaudio CTC forced alignment API](https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)
- [pyopenjtalk](https://github.com/r9y9/pyopenjtalk)
- [Enhancing GOP in CTC-Based Mispronunciation Detection, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/parikh25_interspeech.pdf)
