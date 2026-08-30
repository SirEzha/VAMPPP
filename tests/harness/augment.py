"""Audio and text augmentation utilities for M4 validation harness."""

import io
import re
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch


def add_additive_noise(
    wav: np.ndarray,
    snr_db: float = 20.0,
    noise_type: str = "white",
) -> np.ndarray:
    """Add additive noise (white or pink) to audio at specified SNR (dB)."""
    if len(wav) == 0:
        return wav

    sig_power = np.mean(wav ** 2) + 1e-12
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))

    if noise_type == "white":
        noise = np.random.normal(0, np.sqrt(noise_power), len(wav))
    elif noise_type == "pink":
        # Approximate pink noise via 1/f filtering
        white = np.random.normal(0, 1.0, len(wav))
        fft_w = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(len(wav))
        freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
        pink_fft = fft_w / np.sqrt(freqs)
        noise = np.fft.irfft(pink_fft, n=len(wav))
        curr_p = np.mean(noise ** 2) + 1e-12
        noise = noise * np.sqrt(noise_power / curr_p)
    else:
        noise = np.random.normal(0, np.sqrt(noise_power), len(wav))

    noisy_wav = wav + noise
    max_val = np.max(np.abs(noisy_wav))
    if max_val > 1.0:
        noisy_wav = noisy_wav / max_val
    return noisy_wav.astype(np.float32)


def add_synthetic_reverb(
    wav: np.ndarray,
    sr: int = 16000,
    rt60: float = 0.25,
) -> np.ndarray:
    """Add synthetic exponential decay room impulse response (reverb)."""
    if len(wav) == 0:
        return wav

    rir_len = int(sr * rt60)
    t = np.linspace(0, rt60, rir_len)
    # Exponential decay with Gaussian noise
    decay = np.exp(-3.0 * np.log(10.0) * t / rt60)
    rir = np.random.normal(0, 1.0, rir_len) * decay
    # Add direct path
    rir[0] = 1.0
    rir = rir / (np.sum(np.abs(rir)) + 1e-8)

    reverbed = np.convolve(wav, rir, mode="full")[:len(wav)]
    max_val = np.max(np.abs(reverbed))
    if max_val > 1.0:
        reverbed = reverbed / max_val
    return reverbed.astype(np.float32)


def apply_eq_tilt(
    wav: np.ndarray,
    sr: int = 16000,
    tilt_type: str = "telephone",
) -> np.ndarray:
    """Apply EQ tilt or band-limiting (e.g. telephone bandpass 300-3400 Hz, lowpass, highpass)."""
    if len(wav) == 0:
        return wav

    if tilt_type == "telephone":
        # Bandpass ~300 Hz to 3400 Hz via STFT
        stft = librosa.stft(wav, n_fft=512, hop_length=128)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        mask = (freqs >= 300) & (freqs <= 3400)
        stft[~mask, :] *= 0.05
        filtered = librosa.istft(stft, hop_length=128, length=len(wav))
    elif tilt_type == "lowpass":
        # Lowpass filter at 2000 Hz
        stft = librosa.stft(wav, n_fft=512, hop_length=128)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        mask = freqs > 2000
        stft[mask, :] *= 0.1
        filtered = librosa.istft(stft, hop_length=128, length=len(wav))
    elif tilt_type == "highpass":
        # Highpass filter at 800 Hz
        stft = librosa.stft(wav, n_fft=512, hop_length=128)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
        mask = freqs < 800
        stft[mask, :] *= 0.1
        filtered = librosa.istft(stft, hop_length=128, length=len(wav))
    else:
        filtered = wav

    max_val = np.max(np.abs(filtered))
    if max_val > 1.0:
        filtered = filtered / max_val
    return filtered.astype(np.float32)


def apply_codec_compression(
    wav: np.ndarray,
    sr: int = 16000,
    format: str = "OGG",
) -> np.ndarray:
    """Simulate lossy audio codec encoding/decoding."""
    if len(wav) == 0:
        return wav

    buf = io.BytesIO()
    try:
        # Write to in-memory compressed stream
        sf.write(buf, wav, sr, format=format)
        buf.seek(0)
        decoded, _ = sf.read(buf)
        return decoded.astype(np.float32)
    except Exception:
        # Fallback to quantized bit-depth simulation
        quantized = np.round(wav * 127.0) / 127.0
        return quantized.astype(np.float32)


def shorten_long_vowels(text: str) -> str:
    """Shorten Japanese long vowels (e.g. 'おばあさん' -> 'おばさん', 'とうきょう' -> 'ときょ')."""
    # Replace elongated vowels
    replacements = [
        ("ああ", "あ"),
        ("いい", "い"),
        ("うう", "う"),
        ("ええ", "え"),
        ("おお", "お"),
        ("おう", "お"),
        ("ばあ", "ば"),
        ("じい", "じ"),
        ("とう", "と"),
        ("きょう", "きょ"),
        ("っ", ""),
    ]
    res = text
    for p, r in replacements:
        res = res.replace(p, r)
    return res


def remove_geminates(text: str) -> str:
    """Remove Japanese geminate sokuon 'っ' (e.g. 'がっこう' -> 'がこう', 'きっぷ' -> 'きぷ')."""
    return text.replace("っ", "").replace("ッ", "")


def mutate_phone_substitutions(text_kana: str, k: int = 1) -> str:
    """Inject k phone substitutions into kana text (e.g. /r/ -> /l/ or consonant swaps)."""
    sub_map = {
        "ら": "だ",
        "り": "ぎ",
        "る": "づ",
        "れ": "で",
        "ろ": "ど",
        "か": "た",
        "き": "ち",
        "く": "つ",
        "け": "て",
        "こ": "と",
        "さ": "しゃ",
        "し": "ひ",
        "す": "ふ",
        "せ": "へ",
        "そ": "ほ",
        "な": "ま",
        "に": "み",
        "ぬ": "む",
        "ね": "め",
        "の": "も",
        "は": "わ",
        "ひ": "い",
        "ふ": "う",
        "へ": "え",
        "ほ": "お",
    }

    chars = list(text_kana)
    substitutions_done = 0

    for i in range(len(chars)):
        c = chars[i]
        if c in sub_map and substitutions_done < k:
            chars[i] = sub_map[c]
            substitutions_done += 1

    return "".join(chars)
