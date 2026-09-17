"""Minimalistic, high-performance Pronunciation Similarity Scoring Application.

Compatible with Windows, Linux, and macOS.
Zero bloat: Uses standard Python tkinter + ttk and sounddevice.
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.api import PronunciationScorer, ScoreResult, get_encoder, score
from src.audio import load_audio

# The scoring pipeline always runs at 16 kHz, but the sound card need not accept it.
# PortAudio talks to raw ALSA hw: devices, which expose no resampler, so an
# unsupported rate raises paInvalidSampleRate rather than being converted. Device
# I/O therefore runs at a hardware rate and is resampled on the way in and out.
MODEL_SR = 16000
_CANDIDATE_RATES = (48000, 44100, 32000, 22050, 16000)


def pick_device_samplerate(device: Optional[int], kind: str) -> int:
    """Return a sample rate `device` actually supports, preferring its own default."""
    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    candidates = []
    try:
        candidates.append(int(sd.query_devices(device, kind)["default_samplerate"]))
    except Exception:
        pass
    candidates.extend(_CANDIDATE_RATES)

    for rate in dict.fromkeys(candidates):
        try:
            check(device=device, samplerate=rate, channels=1)
            return rate
        except Exception:
            continue
    raise RuntimeError(f"No supported sample rate for {kind} device {device!r}")


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono float32 audio, passing it through untouched when rates match."""
    if orig_sr == target_sr:
        return wav.astype(np.float32)
    return librosa.resample(
        wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
    ).astype(np.float32)


class VoiceSimilarityApp:
    """Main Application Window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pronunciation Similarity Scorer")
        self.root.geometry("780x820")
        self.root.minsize(680, 700)

        # Apply clean modern styling
        self._configure_styles()

        # Engine & state
        self.scorer: Optional[PronunciationScorer] = None
        self.ref_audio_path: Optional[str] = None
        self.live_audio_data: Optional[np.ndarray] = None
        self.is_recording = False
        self.record_sr = MODEL_SR
        self.recording_thread: Optional[threading.Thread] = None
        self.recorded_chunks: List[np.ndarray] = []
        self.record_start_time: float = 0.0
        self.input_devices = []

        # Build UI layout
        self._build_ui()

        # Populate audio devices
        self.refresh_audio_devices()

        # Lazy load model in background so UI starts instantly
        self.status_var.set("Loading acoustic model in background...")
        threading.Thread(target=self._init_engine, daemon=True).start()

    def _configure_styles(self):
        """Set up modern ttk styles."""
        self.style = ttk.Style()
        try:
            if "clam" in self.style.theme_names():
                self.style.theme_use("clam")
        except Exception:
            pass

        bg_color = "#f7f9fa"
        card_bg = "#ffffff"
        primary_color = "#1a73e8"
        text_color = "#202124"

        self.root.configure(bg=bg_color)
        self.style.configure(".", background=bg_color, foreground=text_color, font=("Segoe UI", 10))
        self.style.configure("Card.TFrame", background=card_bg, relief="flat")
        self.style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), background=card_bg, foreground="#1f2937")
        self.style.configure("Sub.TLabel", font=("Segoe UI", 9), background=card_bg, foreground="#6b7280")
        self.style.configure("Score.TLabel", font=("Segoe UI", 32, "bold"), background=card_bg, foreground=primary_color)
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=6)
        self.style.configure("Record.TButton", font=("Segoe UI", 11, "bold"), foreground="#dc2626", padding=8)

    def _build_ui(self):
        """Construct application widgets."""
        main_container = ttk.Frame(self.root, padding=16)
        main_container.pack(fill=tk.BOTH, expand=True)

        # ----------------------------------------------------
        # 1. Reference Audio & Text Card
        # ----------------------------------------------------
        ref_card = ttk.Frame(main_container, style="Card.TFrame", padding=14)
        ref_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(ref_card, text="1. Native Reference Line", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 6)
        )

        # File selection row
        ttk.Label(ref_card, text="Audio File:", style="Sub.TLabel").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.ref_path_var = tk.StringVar(value="No file loaded")
        ref_path_entry = ttk.Entry(ref_card, textvariable=self.ref_path_var, state="readonly", width=46)
        ref_path_entry.grid(row=1, column=1, sticky=tk.EW, padx=6, pady=2)

        btn_box = ttk.Frame(ref_card, style="Card.TFrame")
        btn_box.grid(row=1, column=2, sticky=tk.E)
        ttk.Button(btn_box, text="Browse...", command=self.browse_reference_file).pack(side=tk.LEFT, padx=2)
        self.btn_play_ref = ttk.Button(btn_box, text="▶ Play", command=self.play_reference_audio, state="disabled")
        self.btn_play_ref.pack(side=tk.LEFT, padx=2)

        # Reference text row
        ttk.Label(ref_card, text="Transcript:", style="Sub.TLabel").grid(row=2, column=0, sticky=tk.W, pady=4)
        self.ref_text_var = tk.StringVar(value="")
        self.ref_text_entry = ttk.Entry(ref_card, textvariable=self.ref_text_var, width=46)
        self.ref_text_entry.grid(row=2, column=1, sticky=tk.EW, padx=6, pady=4)

        # Language selection & options row
        opt_box = ttk.Frame(ref_card, style="Card.TFrame")
        opt_box.grid(row=3, column=1, columnspan=2, sticky=tk.W, pady=(2, 0))

        ttk.Label(opt_box, text="Language:", style="Sub.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        self.lang_var = tk.StringVar(value="ja")
        lang_combo = ttk.Combobox(opt_box, textvariable=self.lang_var, values=["ja", "en-us", "sm"], width=7, state="readonly")
        lang_combo.pack(side=tk.LEFT, padx=(0, 14))

        self.use_f0_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt_box, text="Pitch Accent (F0)", variable=self.use_f0_var).pack(side=tk.LEFT)

        ref_card.columnconfigure(1, weight=1)

        # ----------------------------------------------------
        # 2. Microphone & Live Recording Card
        # ----------------------------------------------------
        mic_card = ttk.Frame(main_container, style="Card.TFrame", padding=14)
        mic_card.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(mic_card, text="2. Live Microphone Input", style="Header.TLabel").grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 6)
        )

        # Mic dropdown
        ttk.Label(mic_card, text="Microphone:", style="Sub.TLabel").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(mic_card, textvariable=self.mic_var, state="readonly", width=42)
        self.mic_combo.grid(row=1, column=1, sticky=tk.EW, padx=6, pady=2)
        ttk.Button(mic_card, text="Refresh", command=self.refresh_audio_devices).grid(row=1, column=2, sticky=tk.W)

        # Recording control row
        rec_box = ttk.Frame(mic_card, style="Card.TFrame")
        rec_box.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=(10, 4))

        self.btn_record = ttk.Button(
            rec_box,
            text="● Start Recording",
            style="Record.TButton",
            command=self.toggle_recording,
        )
        self.btn_record.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_play_live = ttk.Button(rec_box, text="▶ Play Take", command=self.play_live_audio, state="disabled")
        self.btn_play_live.pack(side=tk.LEFT, padx=4)

        ttk.Button(rec_box, text="Load Audio File...", command=self.browse_live_file).pack(side=tk.LEFT, padx=4)

        self.rec_time_var = tk.StringVar(value="")
        ttk.Label(rec_box, textvariable=self.rec_time_var, font=("Segoe UI", 10, "bold"), foreground="#dc2626").pack(
            side=tk.RIGHT, padx=6
        )

        mic_card.columnconfigure(1, weight=1)

        # ----------------------------------------------------
        # 3. Results & Scores Card
        # ----------------------------------------------------
        res_card = ttk.Frame(main_container, style="Card.TFrame", padding=14)
        res_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        ttk.Label(res_card, text="3. Pronunciation Score", style="Header.TLabel").pack(anchor=tk.W)

        # Score display badge
        score_display_box = ttk.Frame(res_card, style="Card.TFrame", padding=10)
        score_display_box.pack(fill=tk.X, pady=4)

        self.headline_score_var = tk.StringVar(value="-- / 100")
        self.score_label = ttk.Label(score_display_box, textvariable=self.headline_score_var, style="Score.TLabel")
        self.score_label.pack(side=tk.LEFT, padx=10)

        # Subscore gauges
        subscores_box = ttk.Frame(score_display_box, style="Card.TFrame")
        subscores_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=20)

        self.gop_label_var = tk.StringVar(value="GOP (Accuracy): --")
        self.dur_label_var = tk.StringVar(value="Duration: --")
        self.dtw_label_var = tk.StringVar(value="PPG-DTW: --")

        ttk.Label(subscores_box, textvariable=self.gop_label_var, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.gop_bar = ttk.Progressbar(subscores_box, length=180, mode="determinate")
        self.gop_bar.pack(fill=tk.X, pady=(1, 4))

        ttk.Label(subscores_box, textvariable=self.dur_label_var, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.dur_bar = ttk.Progressbar(subscores_box, length=180, mode="determinate")
        self.dur_bar.pack(fill=tk.X, pady=(1, 4))

        ttk.Label(subscores_box, textvariable=self.dtw_label_var, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.dtw_bar = ttk.Progressbar(subscores_box, length=180, mode="determinate")
        self.dtw_bar.pack(fill=tk.X, pady=(1, 2))

        # Detailed breakdown treeview table
        table_frame = ttk.Frame(res_card, style="Card.TFrame")
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        columns = ("phone", "start", "end", "duration", "score")
        self.details_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=6)
        self.details_tree.heading("phone", text="Phone (IPA)")
        self.details_tree.heading("start", text="Start (s)")
        self.details_tree.heading("end", text="End (s)")
        self.details_tree.heading("duration", text="Duration (s)")
        self.details_tree.heading("score", text="Phone GOP Score")

        self.details_tree.column("phone", width=100, anchor=tk.CENTER)
        self.details_tree.column("start", width=90, anchor=tk.CENTER)
        self.details_tree.column("end", width=90, anchor=tk.CENTER)
        self.details_tree.column("duration", width=100, anchor=tk.CENTER)
        self.details_tree.column("score", width=130, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.details_tree.yview)
        self.details_tree.configure(yscrollcommand=tree_scroll.set)
        self.details_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ----------------------------------------------------
        # Status Bar
        # ----------------------------------------------------
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9), relief="sunken", anchor=tk.W, padding=(6, 3))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _safe_after(self, func):
        """Invoke a callback on the main GUI thread safely."""
        try:
            if self.root.winfo_exists():
                self.root.after(0, func)
        except Exception:
            pass

    def _init_engine(self):
        """Background initialization of acoustic encoder."""
        try:
            self.scorer = PronunciationScorer()
            self._safe_after(lambda: self.status_var.set("Model ready."))
        except Exception as e:
            msg = str(e)
            self._safe_after(lambda: self.status_var.set(f"Engine init: {msg}"))

    def refresh_audio_devices(self):
        """Query and populate available audio input devices."""
        try:
            devices = sd.query_devices()
            self.input_devices = []
            device_names = []

            for i, d in enumerate(devices):
                if d.get("max_input_channels", 0) > 0:
                    self.input_devices.append((i, d["name"]))
                    device_names.append(f"[{i}] {d['name']}")

            self.mic_combo["values"] = device_names
            if device_names:
                default_in = sd.default.device[0]
                matched_idx = 0
                for idx, (dev_id, _) in enumerate(self.input_devices):
                    if dev_id == default_in:
                        matched_idx = idx
                        break
                self.mic_combo.current(matched_idx)
            else:
                self.mic_combo.set("No microphone found")
        except Exception as e:
            self.mic_combo.set("Error querying audio devices")
            self.status_var.set(f"Audio device error: {e}")

    def browse_reference_file(self):
        """Open file dialog for reference audio."""
        path = filedialog.askopenfilename(
            title="Select Reference Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.ogg *.flac *.m4a"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        self.ref_audio_path = path
        self.ref_path_var.set(os.path.basename(path))
        self.btn_play_ref.configure(state="normal")

        txt_path = Path(path).with_suffix(".txt")
        if txt_path.exists() and not self.ref_text_var.get().strip():
            try:
                content = txt_path.read_text(encoding="utf-8").strip()
                self.ref_text_var.set(content)
            except Exception:
                pass

        self.status_var.set(f"Loaded reference: {os.path.basename(path)}")

    def _play_array(self, wav: np.ndarray, sr: int = MODEL_SR):
        """Play mono audio, resampling to whatever the output device accepts."""
        device = sd.default.device[1]
        out_sr = pick_device_samplerate(device, "output")
        sd.stop()
        sd.play(resample(wav, sr, out_sr), samplerate=out_sr, device=device)

    def play_reference_audio(self):
        """Play selected reference audio file."""
        if not self.ref_audio_path:
            return
        try:
            wav, sr = load_audio(self.ref_audio_path)
            self._play_array(wav, sr)
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio: {e}")

    def play_live_audio(self):
        """Play current live recorded audio."""
        if self.live_audio_data is None:
            return
        try:
            self._play_array(self.live_audio_data, MODEL_SR)
        except Exception as e:
            messagebox.showerror("Playback Error", f"Could not play audio: {e}")

    def browse_live_file(self):
        """Load pre-recorded live take from audio file."""
        path = filedialog.askopenfilename(
            title="Select Live Recording File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.ogg *.flac *.m4a"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            wav, _ = load_audio(path, target_sr=16000)
            self.live_audio_data = wav
            self.btn_play_live.configure(state="normal")
            self.status_var.set(f"Loaded take: {os.path.basename(path)}")
            self.run_scoring_pipeline()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio: {e}")

    def toggle_recording(self):
        """Start or stop microphone recording."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Initiate audio recording on selected device."""
        if not self.ref_audio_path:
            messagebox.showwarning("Missing Reference", "Please load a reference audio file first.")
            return

        if not self.ref_text_var.get().strip():
            messagebox.showwarning("Missing Transcript", "Please enter the reference transcript text.")
            return

        selected_idx = self.mic_combo.current()
        if selected_idx < 0 or selected_idx >= len(self.input_devices):
            dev_id = None
        else:
            dev_id = self.input_devices[selected_idx][0]

        self.recorded_chunks = []
        self.is_recording = True
        self.record_start_time = time.time()
        self.btn_record.configure(text="■ Stop Recording", style="Record.TButton")
        self.status_var.set("Recording in progress...")

        self._update_record_timer()

        self.recording_thread = threading.Thread(
            target=self._record_worker, args=(dev_id,), daemon=True
        )
        self.recording_thread.start()

    def _record_worker(self, device_id: Optional[int]):
        """Stream callback background worker."""
        try:
            self.record_sr = pick_device_samplerate(device_id, "input")
            with sd.InputStream(
                samplerate=self.record_sr,
                channels=1,
                dtype="float32",
                device=device_id,
            ) as stream:
                while self.is_recording:
                    chunk, _ = stream.read(1024)
                    self.recorded_chunks.append(chunk.copy())
        except Exception as e:
            msg = str(e)
            self.is_recording = False
            self._safe_after(lambda: self._on_record_error(msg))

    def _on_record_error(self, msg: str):
        """Surface a capture failure and return the button to its idle state."""
        self.btn_record.configure(text="● Start Recording")
        self.status_var.set("Recording failed.")
        messagebox.showerror("Recording Error", msg)

    def _update_record_timer(self):
        """Update live recording seconds label."""
        if self.is_recording:
            elapsed = int(time.time() - self.record_start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            self.rec_time_var.set(f"● {mins:02d}:{secs:02d}")
            self.root.after(250, self._update_record_timer)
        else:
            self.rec_time_var.set("")

    def stop_recording(self):
        """End recording and launch scoring pipeline."""
        self.is_recording = False
        self.btn_record.configure(text="● Start Recording")
        self.status_var.set("Processing audio...")

        if not self.recorded_chunks:
            self.status_var.set("No audio recorded.")
            return

        full_audio = np.concatenate(self.recorded_chunks, axis=0).flatten()
        self.live_audio_data = resample(full_audio, self.record_sr, MODEL_SR)
        self.btn_play_live.configure(state="normal")

        self.run_scoring_pipeline()

    def run_scoring_pipeline(self):
        """Run pronunciation scoring engine asynchronously."""
        if self.live_audio_data is None or not self.ref_audio_path:
            return

        ref_path = self.ref_audio_path
        ref_text = self.ref_text_var.get().strip()
        live_wav = self.live_audio_data
        lang = self.lang_var.get()
        use_f0 = self.use_f0_var.get()

        self.status_var.set("Scoring pronunciation...")
        threading.Thread(
            target=self._score_worker,
            args=(ref_path, ref_text, live_wav, lang, use_f0),
            daemon=True,
        ).start()

    def _score_worker(self, ref_path, ref_text, live_wav, lang, use_f0):
        """Background scoring computation worker."""
        try:
            t0 = time.time()
            if self.scorer is None:
                self.scorer = PronunciationScorer()

            res: ScoreResult = self.scorer.score(
                reference_audio=ref_path,
                reference_text=ref_text,
                live_audio=live_wav,
                language=lang,
                use_f0=use_f0,
            )
            elapsed = time.time() - t0

            self._safe_after(lambda: self._display_results(res, elapsed))
        except Exception as e:
            err = e
            self._safe_after(lambda: self._on_score_error(err))

    def _display_results(self, res: ScoreResult, elapsed_sec: float):
        """Render computed scores and table details."""
        score_val = res.score
        self.headline_score_var.set(f"{score_val:.1f} / 100")

        if score_val >= 80.0:
            color = "#16a34a"  # Green
        elif score_val >= 60.0:
            color = "#d97706"  # Amber
        else:
            color = "#dc2626"  # Red
        self.score_label.configure(foreground=color)

        self.gop_label_var.set(f"GOP (Accuracy): {res.gop_score:.1f}%")
        self.gop_bar["value"] = res.gop_score

        self.dur_label_var.set(f"Duration (Mora Timing): {res.duration_score:.1f}%")
        self.dur_bar["value"] = res.duration_score

        self.dtw_label_var.set(f"PPG-DTW (Native Flow): {res.dtw_score:.1f}%")
        self.dtw_bar["value"] = res.dtw_score

        for row in self.details_tree.get_children():
            self.details_tree.delete(row)

        for p in res.phone_details:
            sym = p.get("token_str", "?")
            st = f"{p.get('start_sec', 0.0):.2f}"
            en = f"{p.get('end_sec', 0.0):.2f}"
            dur = f"{p.get('end_sec', 0.0) - p.get('start_sec', 0.0):.2f}"
            p_score = f"{p.get('gop_score', 0.0):.1f}%"
            self.details_tree.insert("", tk.END, values=(sym, st, en, dur, p_score))

        self.status_var.set(f"Score: {score_val:.1f} (Completed in {elapsed_sec:.2f}s)")

    def _on_score_error(self, err: Exception):
        """Handle calculation errors gracefully."""
        messagebox.showerror("Scoring Error", f"Failed to score recording:\n{err}")
        self.status_var.set(f"Error: {err}")


def launch():
    """Application entry point."""
    root = tk.Tk()
    app = VoiceSimilarityApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
