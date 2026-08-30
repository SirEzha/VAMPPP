"""Audio preprocessing pipeline.

Identical path for both reference and live audio:
1. Load / decode -> Mono
2. Resample -> 16 kHz
3. Loudness normalize -> -23.0 LUFS
4. Silero VAD -> Trim leading and trailing silence
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import warnings

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad

_VAD_MODEL = None


def get_vad_model():
    """Lazy loader for singleton Silero VAD model."""
    global _VAD_MODEL
    if _VAD_MODEL is None:
        _VAD_MODEL = load_silero_vad()
    return _VAD_MODEL


def load_audio(
    audio_input: Union[str, Path, np.ndarray, torch.Tensor],
    orig_sr: Optional[int] = None,
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load audio from file path, numpy array, or torch tensor.

    Converts to mono float32 in [-1.0, 1.0] and resamples to target_sr.

    Args:
        audio_input: Path to audio file or raw audio array/tensor.
        orig_sr: Original sample rate if audio_input is array/tensor.
        target_sr: Target sample rate (default: 16000).

    Returns:
        Tuple of (audio_1d_numpy, sample_rate).
    """
    if isinstance(audio_input, (str, Path)):
        path = str(audio_input)
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
        return wav.astype(np.float32), target_sr

    if isinstance(audio_input, torch.Tensor):
        wav = audio_input.detach().cpu().float().numpy()
    elif isinstance(audio_input, np.ndarray):
        wav = audio_input.astype(np.float32)
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio_input)}")

    # Handle multi-channel -> Mono
    if wav.ndim > 1:
        if wav.shape[0] <= 8 and wav.shape[0] < wav.shape[1]:
            wav = np.mean(wav, axis=0)
        elif wav.shape[1] <= 8:
            wav = np.mean(wav, axis=1)
        else:
            wav = wav.flatten()

    # Scale int16 ranges if passed as unnormalized floats
    max_abs = np.max(np.abs(wav)) if len(wav) > 0 else 0.0
    if max_abs > 1.0:
        wav = wav / max_abs

    # Resample if needed
    if orig_sr is not None and orig_sr != target_sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)

    return wav.astype(np.float32), target_sr


def normalize_loudness(
    wav: np.ndarray,
    sr: int = 16000,
    target_lufs: float = -23.0,
) -> np.ndarray:
    """Normalize audio loudness to target LUFS using pyloudnorm.

    Falls back to peak normalization if audio is too short or silent.

    Args:
        wav: 1D numpy array of audio samples.
        sr: Sample rate (default: 16000).
        target_lufs: Target loudness in LUFS (default: -23.0).

    Returns:
        Loudness-normalized 1D numpy array.
    """
    if len(wav) == 0:
        return wav

    # Minimum duration required for ITU-R BS.1770 block measurement is ~0.1s
    if len(wav) < int(sr * 0.1):
        # Fallback to peak norm
        peak = np.max(np.abs(wav))
        if peak > 1e-6:
            return (wav / peak * 0.5).astype(np.float32)
        return wav.astype(np.float32)

    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        if np.isneginf(loudness) or np.isnan(loudness):
            # Audio is near silent
            peak = np.max(np.abs(wav))
            if peak > 1e-6:
                return (wav / peak * 0.5).astype(np.float32)
            return wav.astype(np.float32)
        normalized = pyln.normalize.loudness(wav, loudness, target_lufs)
        # Prevent clipping
        peak = np.max(np.abs(normalized))
        if peak > 0.99:
            normalized = normalized / peak * 0.99
        return normalized.astype(np.float32)
    except Exception as e:
        warnings.warn(f"Loudness normalization failed ({e}), using peak normalization.")
        peak = np.max(np.abs(wav))
        if peak > 1e-6:
            return (wav / peak * 0.5).astype(np.float32)
        return wav.astype(np.float32)


def trim_silence_vad(
    wav: np.ndarray,
    sr: int = 16000,
    pad_ms: float = 50.0,
    threshold: float = 0.5,
) -> np.ndarray:
    """Trim leading and trailing silence using Silero VAD.

    Args:
        wav: 1D numpy array of 16 kHz audio.
        sr: Sample rate (must be 16000 or 8000 for Silero VAD).
        pad_ms: Padding in milliseconds to retain around speech onset/offset.
        threshold: Speech probability threshold.

    Returns:
        Trimmed 1D numpy array.
    """
    if len(wav) == 0:
        return wav

    vad = get_vad_model()
    wav_tensor = torch.from_numpy(wav).float()

    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        vad,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=50,
        min_silence_duration_ms=100,
    )

    if not speech_timestamps:
        # If no speech detected with standard threshold, try lower threshold
        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            vad,
            sampling_rate=sr,
            threshold=0.2,
            min_speech_duration_ms=30,
        )

    if not speech_timestamps:
        # Keep full audio if VAD detects nothing (avoid returning empty audio)
        return wav

    pad_samples = int(sr * (pad_ms / 1000.0))
    start_sample = max(0, speech_timestamps[0]["start"] - pad_samples)
    end_sample = min(len(wav), speech_timestamps[-1]["end"] + pad_samples)

    trimmed = wav[start_sample:end_sample]
    if len(trimmed) == 0:
        return wav
    return trimmed.astype(np.float32)


def preprocess_audio(
    audio_input: Union[str, Path, np.ndarray, torch.Tensor],
    orig_sr: Optional[int] = None,
    target_sr: int = 16000,
    target_lufs: float = -23.0,
    vad_pad_ms: float = 50.0,
) -> torch.Tensor:
    """End-to-end preprocessing pipeline for reference and live audio.

    1. Load & convert to mono
    2. Resample to 16 kHz
    3. Normalize loudness to -23.0 LUFS
    4. Trim silence using Silero VAD

    Args:
        audio_input: Audio filepath, numpy array, or torch tensor.
        orig_sr: Sample rate if passing in-memory array/tensor.
        target_sr: Target sample rate (default: 16000).
        target_lufs: Target loudness in LUFS (default: -23.0).
        vad_pad_ms: Padding around speech onset/offset in ms.

    Returns:
        1D torch.FloatTensor of preprocessed 16 kHz audio.
    """
    wav, sr = load_audio(audio_input, orig_sr=orig_sr, target_sr=target_sr)
    wav = normalize_loudness(wav, sr=sr, target_lufs=target_lufs)
    wav = trim_silence_vad(wav, sr=sr, pad_ms=vad_pad_ms)
    return torch.from_numpy(wav).float()
