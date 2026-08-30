"""Pronunciation Similarity Scoring System."""

from src.api import PronunciationScorer, ScoreResult, score
from src.audio import preprocess_audio, load_audio, normalize_loudness, trim_silence_vad
from src.g2p import text_to_phones, phones_to_ids
from src.encoder import AcousticEncoder, get_encoder
from src.align import align_tokens, group_mora_spans, PhoneSpan, MoraSpan
from src.score import (
    compute_phone_gop,
    calibrate_gop_scores,
    compute_duration_score,
    compute_ppg_dtw_score,
    compute_f0_score,
    combine_scores,
)

__all__ = [
    "PronunciationScorer",
    "ScoreResult",
    "score",
    "preprocess_audio",
    "load_audio",
    "normalize_loudness",
    "trim_silence_vad",
    "text_to_phones",
    "phones_to_ids",
    "AcousticEncoder",
    "get_encoder",
    "align_tokens",
    "group_mora_spans",
    "PhoneSpan",
    "MoraSpan",
    "compute_phone_gop",
    "calibrate_gop_scores",
    "compute_duration_score",
    "compute_ppg_dtw_score",
    "compute_f0_score",
    "combine_scores",
]
