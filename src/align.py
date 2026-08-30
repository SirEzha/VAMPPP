"""Forced alignment module using torchaudio CTC forced alignment API."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio.functional as F


@dataclass
class PhoneSpan:
    """Represents an aligned phoneme token span."""

    token_id: int
    token_str: str
    peak_frame: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    score: float

    @property
    def duration_frames(self) -> int:
        return max(1, self.end_frame - self.start_frame)

    @property
    def duration_sec(self) -> float:
        return max(0.001, self.end_sec - self.start_sec)


@dataclass
class MoraSpan:
    """Represents a grouped mora span for Japanese duration calculation."""

    mora_index: int
    phones: List[PhoneSpan]
    mora_str: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float

    @property
    def duration_frames(self) -> int:
        return max(1, self.end_frame - self.start_frame)

    @property
    def duration_sec(self) -> float:
        return max(0.001, self.end_sec - self.start_sec)


# Japanese vowel phonemes in eSpeak-IPA
JA_VOWELS = {"a", "i", "ɯ", "e", "o", "ä", "e̞", "o̞", "ɯᵝ"}


def align_tokens(
    log_probs: torch.Tensor,
    target_ids: List[int],
    inv_vocab: Dict[int, str],
    blank_id: int = 0,
    frame_shift_sec: float = 0.020,  # 20ms per frame for wav2vec2 (16kHz / 320 downsampling)
) -> List[PhoneSpan]:
    """Perform CTC forced alignment on log_probs against target token sequence.

    Constructs contiguous phone intervals between token transitions.

    Args:
        log_probs: (1, T, V) or (T, V) CTC log-probabilities tensor.
        target_ids: List of target integer token IDs.
        inv_vocab: Dictionary mapping token ID to phone string.
        blank_id: Blank token ID (default: 0).
        frame_shift_sec: Time duration per frame in seconds (default: 0.020).

    Returns:
        List of PhoneSpan objects with contiguous segment boundaries.
    """
    if len(target_ids) == 0:
        return []

    if log_probs.ndim == 2:
        lp = log_probs.unsqueeze(0)  # (1, T, V)
    else:
        lp = log_probs

    T_frames = lp.shape[1]
    S_tokens = len(target_ids)

    # Convert log_probs to float32 on same device
    lp_float = lp.float()
    device = lp.device

    # Ensure length constraint
    if T_frames < S_tokens:
        pad_frames = S_tokens - T_frames + 2
        lp_float = torch.nn.functional.pad(lp_float, (0, 0, 0, pad_frames, 0, 0), mode="replicate")
        T_frames = lp_float.shape[1]

    targets = torch.tensor([target_ids], dtype=torch.int32, device=device)
    input_lengths = torch.tensor([T_frames], dtype=torch.int32, device=device)
    target_lengths = torch.tensor([S_tokens], dtype=torch.int32, device=device)

    try:
        aligned_tokens, scores = F.forced_align(
            lp_float,
            targets,
            input_lengths,
            target_lengths,
            blank=blank_id,
        )
        merged = F.merge_tokens(aligned_tokens[0], scores[0])
    except Exception:
        merged = []

    # Extract peak frame and scores for each target token
    raw_spans = []
    if len(merged) == len(target_ids):
        for span in merged:
            raw_spans.append((span.token, span.start, span.end, float(span.score)))
    elif len(merged) > 0:
        # Map available merged tokens
        for i, tid in enumerate(target_ids):
            if i < len(merged):
                span = merged[i]
                raw_spans.append((tid, span.start, span.end, float(span.score)))
            else:
                last_end = raw_spans[-1][2] if raw_spans else 0
                raw_spans.append((tid, last_end, min(T_frames, last_end + 1), -5.0))
    else:
        # Uniform fallback
        step = max(1, T_frames // S_tokens)
        for i, tid in enumerate(target_ids):
            st = i * step
            en = (i + 1) * step if i < S_tokens - 1 else T_frames
            raw_spans.append((tid, st, en, -5.0))

    # Construct contiguous phone segments between consecutive tokens
    peaks = [int((st + en) // 2) for _, st, en, _ in raw_spans]
    boundaries = [0]
    for i in range(len(peaks) - 1):
        mid = (peaks[i] + peaks[i + 1]) // 2
        boundaries.append(mid)
    boundaries.append(T_frames)

    spans: List[PhoneSpan] = []
    for i, (tid, _, _, score_val) in enumerate(raw_spans):
        st = boundaries[i]
        en = max(st + 1, boundaries[i + 1])
        tok_str = inv_vocab.get(tid, f"<{tid}>")
        spans.append(
            PhoneSpan(
                token_id=tid,
                token_str=tok_str,
                peak_frame=peaks[i],
                start_frame=st,
                end_frame=en,
                start_sec=st * frame_shift_sec,
                end_sec=en * frame_shift_sec,
                score=score_val,
            )
        )

    return spans


def group_mora_spans(
    phone_spans: List[PhoneSpan],
    language: str = "ja",
) -> List[MoraSpan]:
    """Group aligned phone spans into linguistic morae for duration evaluation.

    In Japanese:
    - Consonant(s) + Vowel -> 1 mora
    - Standalone Vowel -> 1 mora
    - Moraic nasal ('n' / 'ɴ') -> 1 mora
    - Sokuon (geminate consonant) -> 1 mora

    Args:
        phone_spans: List of PhoneSpan objects.
        language: Language code ('ja', etc.).

    Returns:
        List of MoraSpan objects.
    """
    if not phone_spans:
        return []

    if language.lower() not in ("ja", "jp", "japanese"):
        return [
            MoraSpan(
                mora_index=i,
                phones=[p],
                mora_str=p.token_str,
                start_frame=p.start_frame,
                end_frame=p.end_frame,
                start_sec=p.start_sec,
                end_sec=p.end_sec,
            )
            for i, p in enumerate(phone_spans)
        ]

    morae: List[MoraSpan] = []
    curr_phones: List[PhoneSpan] = []
    mora_idx = 0

    for p in phone_spans:
        curr_phones.append(p)
        sym = p.token_str

        is_vowel = sym in JA_VOWELS
        is_nasal = sym in ("n", "ɴ", "N") and len(curr_phones) == 1

        if is_vowel or is_nasal or len(curr_phones) >= 3:
            st_frame = curr_phones[0].start_frame
            en_frame = curr_phones[-1].end_frame
            mora_str = "".join([x.token_str for x in curr_phones])
            morae.append(
                MoraSpan(
                    mora_index=mora_idx,
                    phones=list(curr_phones),
                    mora_str=mora_str,
                    start_frame=st_frame,
                    end_frame=en_frame,
                    start_sec=curr_phones[0].start_sec,
                    end_sec=curr_phones[-1].end_sec,
                )
            )
            mora_idx += 1
            curr_phones = []

    if curr_phones:
        st_frame = curr_phones[0].start_frame
        en_frame = curr_phones[-1].end_frame
        mora_str = "".join([x.token_str for x in curr_phones])
        morae.append(
            MoraSpan(
                mora_index=mora_idx,
                phones=list(curr_phones),
                mora_str=mora_str,
                start_frame=st_frame,
                end_frame=en_frame,
                start_sec=curr_phones[0].start_sec,
                end_sec=curr_phones[-1].end_sec,
            )
        )

    return morae
