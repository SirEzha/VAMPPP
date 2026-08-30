"""Scoring terms and combination logic.

Terms:
1. GOP (Goodness of Pronunciation): CTC log-posteriors over aligned spans vs reference ceiling.
2. Duration: speech-rate normalized mora durations (live vs reference anchor).
3. PPG-DTW: Sakoe-Chiba banded DTW with symmetric step pattern P=0.5.
4. F0: DTW-aligned correlation of z-scored log-F0 (opt-in).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch

from src.align import MoraSpan, PhoneSpan, group_mora_spans


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of pronunciation subscores."""

    composite_score: float  # Final 0-100 score
    gop_score: float  # GOP score 0-100
    duration_score: float  # Duration score 0-100
    dtw_score: float  # PPG-DTW similarity score 0-100
    f0_score: Optional[float] = None  # F0 pitch accent score (if enabled)
    phone_details: List[Dict[str, Any]] = field(default_factory=list)
    mora_details: List[Dict[str, Any]] = field(default_factory=list)
    dtw_cost: float = 0.0


def compute_phone_gop(
    log_probs: torch.Tensor,
    phone_spans: List[PhoneSpan],
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute per-phone GOP (Goodness of Pronunciation).

    GOP for phone p over span [t_start, t_end):
        GOP(p) = (1 / T_p) * sum_{t=t_start}^{t_end-1} log P(p | x_t)

    Args:
        log_probs: (1, T, V) or (T, V) log-probabilities.
        phone_spans: Aligned PhoneSpan list.

    Returns:
        Tuple of (average_gop, per_phone_details_list).
    """
    if not phone_spans:
        return 0.0, []

    if log_probs.ndim == 3:
        lp = log_probs[0]  # (T, V)
    else:
        lp = log_probs

    lp_np = lp.detach().cpu().float().numpy()
    T_max = lp_np.shape[0]

    phone_details = []
    gop_values = []

    for span in phone_spans:
        st = min(max(0, span.start_frame), T_max - 1)
        en = min(max(st + 1, span.end_frame), T_max)
        tid = span.token_id

        # Average log-posterior of the target token over the span
        span_lps = lp_np[st:en, tid]
        avg_lp = float(np.mean(span_lps))

        gop_values.append(avg_lp)
        phone_details.append(
            {
                "token_id": tid,
                "token_str": span.token_str,
                "start_frame": st,
                "end_frame": en,
                "start_sec": span.start_sec,
                "end_sec": span.end_sec,
                "raw_gop": avg_lp,
            }
        )

    mean_gop = float(np.mean(gop_values)) if gop_values else -10.0
    return mean_gop, phone_details


def calibrate_gop_scores(
    live_phone_details: List[Dict[str, Any]],
    ref_phone_details: List[Dict[str, Any]],
    temperature: float = 2.5,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Calibrate live phone GOP values against the reference native ceiling.

    Normalized score per phone:
        S_gop(p) = 100 * exp( min(0, GOP_live(p) - GOP_ref(p)) / temperature )

    Args:
        live_phone_details: Phone details from live audio.
        ref_phone_details: Phone details from reference audio.
        temperature: Sensitivity temperature (default: 2.5).

    Returns:
        Tuple of (mean_gop_score_0_to_100, enriched_phone_details).
    """
    if not live_phone_details:
        return 0.0, []

    scores = []
    enriched = []

    for i, live_p in enumerate(live_phone_details):
        live_gop = live_p["raw_gop"]
        ref_gop = (
            ref_phone_details[i]["raw_gop"]
            if i < len(ref_phone_details)
            else live_gop
        )

        delta = live_gop - ref_gop
        phone_score = 100.0 * float(np.exp(min(0.0, delta) / temperature))

        entry = dict(live_p)
        entry["ref_gop"] = ref_gop
        entry["gop_score"] = phone_score
        enriched.append(entry)
        scores.append(phone_score)

    mean_score = float(np.mean(scores)) if scores else 0.0
    return mean_score, enriched


def compute_duration_score(
    live_spans: List[PhoneSpan],
    ref_spans: List[PhoneSpan],
    language: str = "ja",
    lambda_dur: float = 2.5,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute speech-rate normalized mora duration score (live vs reference).

    Catches Japanese long-vowel and geminate sokuon duration errors.

    Args:
        live_spans: Aligned live phone spans.
        ref_spans: Aligned reference phone spans.
        language: Target language code ('ja', etc.).
        lambda_dur: Duration penalty scaling coefficient (default: 2.5).

    Returns:
        Tuple of (duration_score_0_to_100, mora_details_list).
    """
    live_morae = group_mora_spans(live_spans, language=language)
    ref_morae = group_mora_spans(ref_spans, language=language)

    if not live_morae or not ref_morae:
        return 100.0, []

    num_units = min(len(live_morae), len(ref_morae))
    live_durs = np.array([m.duration_sec for m in live_morae[:num_units]], dtype=np.float64)
    ref_durs = np.array([m.duration_sec for m in ref_morae[:num_units]], dtype=np.float64)

    mean_live_dur = max(0.01, float(np.mean(live_durs)))
    mean_ref_dur = max(0.01, float(np.mean(ref_durs)))

    norm_live = live_durs / mean_live_dur
    norm_ref = ref_durs / mean_ref_dur

    mora_details = []
    log_errors = []

    for i in range(num_units):
        l_mora = live_morae[i]
        r_mora = ref_morae[i]

        ratio = norm_live[i] / max(1e-4, norm_ref[i])
        log_err = float(np.log(ratio))
        log_errors.append(log_err ** 2)

        mora_score = 100.0 * float(np.exp(-lambda_dur * (log_err ** 2)))
        mora_details.append(
            {
                "mora_index": i,
                "mora_str": l_mora.mora_str,
                "live_duration_sec": float(l_mora.duration_sec),
                "ref_duration_sec": float(r_mora.duration_sec),
                "ratio": float(ratio),
                "mora_score": float(mora_score),
            }
        )

    mean_sq_log_err = float(np.mean(log_errors)) if log_errors else 0.0
    dur_score = 100.0 * float(np.exp(-lambda_dur * mean_sq_log_err))
    return dur_score, mora_details


def sakoe_chiba_dtw(
    x: np.ndarray,
    y: np.ndarray,
    band_ratio: float = 0.35,
    return_path: bool = True,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Symmetric P=0.5 Dynamic Time Warping with Sakoe-Chiba band.

    Symmetric step pattern P=0.5 prevents flat / degenerate alignments.

    Args:
        x: (T1, D) feature sequence (live features).
        y: (T2, D) feature sequence (ref features).
        band_ratio: Sakoe-Chiba band width ratio relative to sequence length.
        return_path: Whether to return the optimal alignment path.

    Returns:
        Normalized path cost, and optionally the (N, 2) alignment path array.
    """
    T1, D = x.shape
    T2, _ = y.shape

    if T1 == 0 or T2 == 0:
        if return_path:
            return 0.0, np.zeros((0, 2), dtype=np.int32)
        return 0.0

    # Row-wise L2 normalization for cosine similarity
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    y_norm = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-8)
    sim = np.dot(x_norm, y_norm.T)
    dist = np.clip(1.0 - sim, 0.0, 2.0).astype(np.float64)

    length_diff = abs(T1 - T2)
    band_r = max(5, length_diff + 5, int(band_ratio * max(T1, T2)))

    cost = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    backtrack = np.zeros((T1 + 1, T2 + 1), dtype=np.int8)

    slope = T2 / float(T1)
    for i in range(1, T1 + 1):
        center_j = int(round(i * slope))
        j_min = max(1, center_j - band_r)
        j_max = min(T2, center_j + band_r)
        d_i = dist[i - 1]

        for j in range(j_min, j_max + 1):
            d_curr = d_i[j - 1]

            # 1. Diag (i-1, j-1)
            c1 = cost[i - 1, j - 1] + 2.0 * d_curr
            # 2. (i-1, j-2)
            c2 = cost[i - 1, j - 2] + 2.0 * dist[i - 1, j - 2] + d_curr if j >= 2 else np.inf
            # 3. (i-2, j-1)
            c3 = cost[i - 2, j - 1] + 2.0 * dist[i - 2, j - 1] + d_curr if i >= 2 else np.inf
            # 4. Standard step fallback if disconnected
            c4 = cost[i - 1, j] + d_curr
            c5 = cost[i, j - 1] + d_curr

            best_c = c1
            best_step = 1
            if c2 < best_c:
                best_c = c2
                best_step = 2
            if c3 < best_c:
                best_c = c3
                best_step = 3
            if not np.isfinite(best_c):
                if c4 < best_c:
                    best_c = c4
                    best_step = 4
                if c5 < best_c:
                    best_c = c5
                    best_step = 5

            cost[i, j] = best_c
            backtrack[i, j] = best_step

    if not np.isfinite(cost[T1, T2]):
        cost.fill(np.inf)
        cost[0, 0] = 0.0
        backtrack.fill(1)
        for i in range(1, T1 + 1):
            for j in range(1, T2 + 1):
                d_curr = dist[i - 1, j - 1]
                c_diag = cost[i - 1, j - 1] + 2.0 * d_curr
                c_up = cost[i - 1, j] + d_curr
                c_left = cost[i, j - 1] + d_curr
                m = min(c_diag, c_up, c_left)
                cost[i, j] = m
                backtrack[i, j] = 1 if m == c_diag else (4 if m == c_up else 5)

    norm_cost = float(cost[T1, T2] / max(1, T1 + T2))

    if not return_path:
        return norm_cost

    # Backtrack path
    path = []
    i, j = T1, T2
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = backtrack[i, j]
        if step == 1:
            i -= 1
            j -= 1
        elif step == 2:
            path.append((i - 1, j - 2))
            i -= 1
            j -= 2
        elif step == 3:
            path.append((i - 2, j - 1))
            i -= 2
            j -= 1
        elif step == 4:
            i -= 1
        elif step == 5:
            j -= 1
        else:
            i -= 1
            j -= 1

    path.reverse()
    return norm_cost, np.array(path, dtype=np.int32)


def compute_ppg_dtw_score(
    live_features: np.ndarray,
    ref_features: np.ndarray,
    band_ratio: float = 0.35,
    lambda_dtw: float = 2.0,
) -> Tuple[float, float, np.ndarray]:
    """Compute DTW similarity score on non-blank PPG / feature representations.

    Args:
        live_features: (T1, D) live representations.
        ref_features: (T2, D) reference representations.
        band_ratio: Sakoe-Chiba band width ratio (default: 0.35).
        lambda_dtw: Distance scaling parameter (default: 2.0).

    Returns:
        Tuple of (dtw_similarity_score_0_to_100, norm_cost, alignment_path).
    """
    cost, path = sakoe_chiba_dtw(live_features, ref_features, band_ratio=band_ratio, return_path=True)
    sim_score = 100.0 * float(np.exp(-lambda_dtw * cost))
    return sim_score, cost, path


def compute_f0_score(
    live_wav: np.ndarray,
    ref_wav: np.ndarray,
    dtw_path: np.ndarray,
    sr: int = 16000,
    hop_length: int = 320,  # 20ms frame shift matching wav2vec2
) -> float:
    """Compute DTW-aligned correlation of z-scored log-F0 contours (pitch accent term).

    Gender/speaker invariant: per-utterance z-score removes absolute pitch level.

    Args:
        live_wav: 16 kHz live audio array.
        ref_wav: 16 kHz ref audio array.
        dtw_path: (N, 2) DTW alignment path.
        sr: Sample rate (default: 16000).
        hop_length: Hop length matching acoustic encoder (default: 320).

    Returns:
        F0 similarity score in [0.0, 100.0].
    """
    if len(dtw_path) == 0 or len(live_wav) < hop_length or len(ref_wav) < hop_length:
        return 100.0

    try:
        f0_live, voiced_live, _ = librosa.pyin(
            live_wav, fmin=50, fmax=500, sr=sr, hop_length=hop_length
        )
        f0_ref, voiced_ref, _ = librosa.pyin(
            ref_wav, fmin=50, fmax=500, sr=sr, hop_length=hop_length
        )

        f0_live = np.nan_to_num(f0_live, nan=0.0)
        f0_ref = np.nan_to_num(f0_ref, nan=0.0)

        log_live = np.zeros_like(f0_live)
        mask_live = f0_live > 0
        if np.any(mask_live):
            log_live[mask_live] = np.log(f0_live[mask_live])
            mean_l = np.mean(log_live[mask_live])
            std_l = np.std(log_live[mask_live]) + 1e-6
            log_live[mask_live] = (log_live[mask_live] - mean_l) / std_l

        log_ref = np.zeros_like(f0_ref)
        mask_ref = f0_ref > 0
        if np.any(mask_ref):
            log_ref[mask_ref] = np.log(f0_ref[mask_ref])
            mean_r = np.mean(log_ref[mask_ref])
            std_r = np.std(log_ref[mask_ref]) + 1e-6
            log_ref[mask_ref] = (log_ref[mask_ref] - mean_r) / std_r

        vals_live = []
        vals_ref = []
        for i_l, i_r in dtw_path:
            idx_l = min(i_l, len(log_live) - 1)
            idx_r = min(i_r, len(log_ref) - 1)
            if mask_live[idx_l] and mask_ref[idx_r]:
                vals_live.append(log_live[idx_l])
                vals_ref.append(log_ref[idx_r])

        if len(vals_live) < 5:
            return 100.0

        v_l = np.array(vals_live)
        v_r = np.array(vals_ref)

        norm_prod = (np.linalg.norm(v_l) * np.linalg.norm(v_r)) + 1e-7
        corr = float(np.dot(v_l, v_r) / norm_prod)
        score = 100.0 * max(0.0, corr)
        return float(np.clip(score, 0.0, 100.0))
    except Exception:
        return 100.0


def combine_scores(
    gop_score: float,
    dur_score: float,
    dtw_score: float,
    f0_score: Optional[float] = None,
    use_f0: bool = False,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Combine subscores into a single calibrated 0-100 float score.

    Default term weights:
    - GOP (primary): 0.55
    - Duration: 0.25
    - PPG-DTW: 0.20
    (If F0 enabled: 0.45, 0.20, 0.20, 0.15)

    Args:
        gop_score: GOP score in [0, 100].
        dur_score: Duration score in [0, 100].
        dtw_score: PPG-DTW score in [0, 100].
        f0_score: Optional F0 score in [0, 100].
        use_f0: Whether F0 is included in combination.
        weights: Optional custom weight dictionary.

    Returns:
        Composite score in [0.0, 100.0].
    """
    if weights is None:
        if use_f0 and f0_score is not None:
            w = {"gop": 0.45, "dur": 0.20, "dtw": 0.20, "f0": 0.15}
        else:
            w = {"gop": 0.55, "dur": 0.25, "dtw": 0.20}
    else:
        w = weights

    total_w = sum(w.values())
    raw_comp = (
        w.get("gop", 0.0) * gop_score
        + w.get("dur", 0.0) * dur_score
        + w.get("dtw", 0.0) * dtw_score
    )

    if use_f0 and f0_score is not None and "f0" in w:
        raw_comp += w["f0"] * f0_score

    final_score = raw_comp / max(1e-6, total_w)
    return float(np.clip(final_score, 0.0, 100.0))
