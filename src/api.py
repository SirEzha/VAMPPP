"""Public API for Pronunciation Similarity Scoring."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from src.align import PhoneSpan, align_tokens
from src.audio import preprocess_audio
from src.encoder import AcousticEncoder, get_encoder
from src.g2p import phones_to_ids, text_to_phones
from src.score import (
    ScoreBreakdown,
    calibrate_gop_scores,
    combine_scores,
    compute_duration_score,
    compute_f0_score,
    compute_phone_gop,
    compute_ppg_dtw_score,
)


@dataclass
class ScoreResult:
    """Pronunciation similarity scoring output object."""

    score: float  # Single headline score in [0.0, 100.0]
    gop_score: float  # Goodness of Pronunciation subscore [0.0, 100.0]
    duration_score: float  # Speech-rate normalized duration subscore [0.0, 100.0]
    dtw_score: float  # Length-normalized PPG-DTW similarity [0.0, 100.0]
    f0_score: Optional[float] = None  # Pitch accent correlation score (if enabled)
    phone_details: List[Dict[str, Any]] = field(default_factory=list)
    mora_details: List[Dict[str, Any]] = field(default_factory=list)
    ref_phone_details: List[Dict[str, Any]] = field(default_factory=list)
    dtw_cost: float = 0.0

    def __float__(self) -> float:
        return float(self.score)

    def __int__(self) -> int:
        return int(round(self.score))

    def __repr__(self) -> str:
        f0_str = f", f0={self.f0_score:.2f}" if self.f0_score is not None else ""
        return (
            f"<ScoreResult score={self.score:.2f} (GOP={self.gop_score:.2f}, "
            f"Dur={self.duration_score:.2f}, DTW={self.dtw_score:.2f}{f0_str})>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CachedReference:
    """Pre-computed representations for a reference anchor line."""

    text: str
    language: str
    phones: List[str]
    target_ids: List[int]
    wav_tensor: torch.Tensor
    wav_np: np.ndarray
    log_probs: torch.Tensor
    ppg_np: np.ndarray
    phone_spans: List[PhoneSpan]
    phone_details: List[Dict[str, Any]]


class PronunciationScorer:
    """Main zero-shot pronunciation scoring engine."""

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        device: Optional[str] = None,
    ):
        self.encoder = get_encoder(model_id=model_id, device=device)
        self.ref_cache: Dict[str, CachedReference] = {}

    def prepare_reference(
        self,
        reference_audio: Union[str, Path, np.ndarray, torch.Tensor],
        reference_text: str,
        language: str = "ja",
        kana_override: Optional[str] = None,
    ) -> CachedReference:
        """Pre-compute and cache reference anchor line representation."""
        cache_key = f"{reference_text}_{language}_{kana_override}"
        if cache_key in self.ref_cache:
            return self.ref_cache[cache_key]

        # 1. Preprocess reference audio
        ref_wav_t = preprocess_audio(reference_audio)
        ref_wav_np = ref_wav_t.cpu().numpy()

        # 2. G2P conversion
        phones = text_to_phones(reference_text, language=language, kana_override=kana_override)
        valid_phones, target_ids = phones_to_ids(phones, self.encoder.vocab)

        # 3. Acoustic encoding (PPG)
        enc_out = self.encoder.extract(ref_wav_t)
        ref_lp = enc_out.log_probs
        # Non-blank softmax PPG (speaker-normalized phone posteriors)
        ref_ppg_np = torch.softmax(enc_out.logits[0, :, 1:], dim=-1).detach().cpu().float().numpy()

        # 4. Forced alignment
        ref_spans = align_tokens(
            ref_lp,
            target_ids=target_ids,
            inv_vocab=self.encoder.inv_vocab,
            blank_id=self.encoder.blank_id,
        )

        # 5. Reference GOP ceiling
        _, ref_phone_details = compute_phone_gop(ref_lp, ref_spans)

        cached = CachedReference(
            text=reference_text,
            language=language,
            phones=valid_phones,
            target_ids=target_ids,
            wav_tensor=ref_wav_t,
            wav_np=ref_wav_np,
            log_probs=ref_lp,
            ppg_np=ref_ppg_np,
            phone_spans=ref_spans,
            phone_details=ref_phone_details,
        )
        self.ref_cache[cache_key] = cached
        return cached

    def score(
        self,
        reference_audio: Union[str, Path, np.ndarray, torch.Tensor],
        reference_text: str,
        live_audio: Union[str, Path, np.ndarray, torch.Tensor],
        language: str = "ja",
        kana_override: Optional[str] = None,
        use_f0: bool = False,
        weights: Optional[Dict[str, float]] = None,
    ) -> ScoreResult:
        """Score live recording against reference audio and text.

        Args:
            reference_audio: Path or array/tensor of native reference audio.
            reference_text: Known transcript text of the reference line.
            live_audio: Path or array/tensor of live microphone recording.
            language: Target language code ('ja', 'en-us', etc.).
            kana_override: Optional kana reading for ambiguous Japanese kanji.
            use_f0: Whether to enable pitch accent F0 correlation term.
            weights: Optional custom dictionary of term weights.

        Returns:
            ScoreResult with headline score (0-100) and subscore breakdown.
        """
        # 1. Reference Anchor
        ref = self.prepare_reference(
            reference_audio=reference_audio,
            reference_text=reference_text,
            language=language,
            kana_override=kana_override,
        )

        # 2. Preprocess live audio (identical path)
        live_wav_t = preprocess_audio(live_audio)
        live_wav_np = live_wav_t.cpu().numpy()

        # 3. Acoustic encoding on live audio
        enc_out = self.encoder.extract(live_wav_t)
        live_lp = enc_out.log_probs
        live_ppg_np = torch.softmax(enc_out.logits[0, :, 1:], dim=-1).detach().cpu().float().numpy()

        # 4. Forced alignment on live audio (using identical target phone sequence)
        live_spans = align_tokens(
            live_lp,
            target_ids=ref.target_ids,
            inv_vocab=self.encoder.inv_vocab,
            blank_id=self.encoder.blank_id,
        )

        # 5. Term 1: GOP (Goodness of Pronunciation, calibrated to reference)
        _, raw_live_phone_details = compute_phone_gop(live_lp, live_spans)
        gop_score, calibrated_phones = calibrate_gop_scores(
            live_phone_details=raw_live_phone_details,
            ref_phone_details=ref.phone_details,
        )

        # 6. Term 2: Duration score (speech-rate normalized mora comparison)
        dur_score, mora_details = compute_duration_score(
            live_spans=live_spans,
            ref_spans=ref.phone_spans,
            language=language,
        )

        # 7. Term 3: DTW score on non-blank PPG (Sakoe-Chiba banded DTW)
        dtw_score, dtw_cost, dtw_path = compute_ppg_dtw_score(
            live_features=live_ppg_np,
            ref_features=ref.ppg_np,
        )

        # 8. Term 4: F0 score (opt-in)
        f0_score: Optional[float] = None
        if use_f0:
            f0_score = compute_f0_score(
                live_wav=live_wav_np,
                ref_wav=ref.wav_np,
                dtw_path=dtw_path,
            )

        # 9. Composite score combination
        final_score = combine_scores(
            gop_score=gop_score,
            dur_score=dur_score,
            dtw_score=dtw_score,
            f0_score=f0_score,
            use_f0=use_f0,
            weights=weights,
        )

        return ScoreResult(
            score=final_score,
            gop_score=gop_score,
            duration_score=dur_score,
            dtw_score=dtw_score,
            f0_score=f0_score,
            phone_details=calibrated_phones,
            mora_details=mora_details,
            ref_phone_details=ref.phone_details,
            dtw_cost=dtw_cost,
        )


_SCORER_SINGLETON: Optional[PronunciationScorer] = None


def score(
    reference_audio: Union[str, Path, np.ndarray, torch.Tensor],
    reference_text: str,
    live_audio: Union[str, Path, np.ndarray, torch.Tensor],
    language: str = "ja",
    kana_override: Optional[str] = None,
    use_f0: bool = False,
    weights: Optional[Dict[str, float]] = None,
    device: Optional[str] = None,
) -> ScoreResult:
    """Convenience function to score a live recording against a reference.

    Args:
        reference_audio: Native reference audio file path or array.
        reference_text: Known reference line text.
        live_audio: Live recording file path or array.
        language: Language code (default: 'ja').
        kana_override: Optional kana/furigana text.
        use_f0: Whether to enable pitch accent term (default: False).
        weights: Optional custom term weights dictionary.
        device: Device to use ('cuda' or 'cpu').

    Returns:
        ScoreResult with headline score (0-100) and subscores.
    """
    global _SCORER_SINGLETON
    if _SCORER_SINGLETON is None or (_SCORER_SINGLETON.encoder.device != device and device is not None):
        _SCORER_SINGLETON = PronunciationScorer(device=device)

    return _SCORER_SINGLETON.score(
        reference_audio=reference_audio,
        reference_text=reference_text,
        live_audio=live_audio,
        language=language,
        kana_override=kana_override,
        use_f0=use_f0,
        weights=weights,
    )
