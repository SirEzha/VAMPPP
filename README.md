# Pronunciation Similarity Scoring Application

A lightweight, zero-shot pronunciation similarity scoring desktop app and Python engine.

---

## Key Features

- **Zero-Shot Scoring**: No training data or labeled pronunciation ratings needed.
- **Reference-Anchored Calibration**: Scores live microphone recordings relative to the native reference ceiling.
- **Minimal Footprint**: Native Python GUI (`tkinter` + `ttk`), zero webview/Electron bloat.
- **Cross-Platform**: Fully compatible with **Windows**, **Linux**, and **macOS**.
- **Comprehensive Pronunciation Subscores**:
  - **GOP (Goodness of Pronunciation)**: Phone-level acoustic posterior evaluation.
  - **Mora Duration**: Catch Japanese long-vowel (`おばあさん` vs `おばさん`) and geminate sokuon (`がっこう` vs `がこう`) errors.
  - **PPG-DTW**: Sakoe-Chiba banded Dynamic Time Warping on non-blank phone posteriors.
  - **Pitch Accent (F0)**: Opt-in correlation of z-scored log-F0 contours.

---

## Installation

### 1. Conda Environment
```bash
conda env create -f environment.yml
conda activate voice_similarity
```

Or install dependencies using `pip`:
```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the desktop application with:

```bash
python app.py
```

### Application Controls:
1. **Reference Audio**: Click `Browse...` to select a reference `.wav`/`.mp3`/`.ogg` file. If a matching `.txt` transcript file exists, it will auto-populate.
2. **Microphone Selection**: Choose your input microphone from the dropdown list.
3. **Record / Test**:
   - Click `● Start Recording`, speak the phrase into the microphone, then click `■ Stop Recording`.
   - Or click `Load Audio File...` to evaluate a pre-recorded take.
4. **Instant Score & Breakdown**:
   - **Headline Score**: 0–100 calibrated pronunciation rating with color badge.
   - **Subscore Progress Bars**: GOP, Mora Duration, and PPG-DTW.
   - **Phoneme Table**: Detailed per-phone IPA alignment, start/end timestamps, and accuracy scores.

---

## Python API Usage

```python
from src import score

# Score live audio against a native reference
result = score(
    reference_audio="data/refs/sample_ja.wav",
    reference_text="こんにちは、世界",
    live_audio="path/to/live_recording.wav",
    language="ja",
    use_f0=False,
)

print(f"Headline Score: {float(result):.2f}/100")
print(f"GOP Score: {result.gop_score:.2f}%")
print(f"Duration Score: {result.duration_score:.2f}%")
print(f"DTW Score: {result.dtw_score:.2f}%")

# Inspect per-phone IPA breakdown
for p in result.phone_details:
    print(f"Phone {p['token_str']:5s}: {p['gop_score']:.1f}% ({p['start_sec']:.2f}s - {p['end_sec']:.2f}s)")
```

---

## Running Test Suite

```bash
pytest -v
```
