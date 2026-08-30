"""M4 Validation Harness Probe Table.

Validates the 5 essential quality gates for zero-shot pronunciation scoring:
1. Speaker Invariance: same phrase, multiple voices/genders -> low correlation with ECAPA distance.
2. Channel Invariance: audio augmentations (reverb, noise, EQ, codec) -> small score drop.
3. Duration Sensitivity: phonemic duration violations (long vowels, geminates) -> material score drop.
4. Monotonicity: k phone mutations -> monotonic score decrease in k.
5. Discrimination: completely different phrase -> score floors.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pyopenjtalk
import pytest
import torch
from speechbrain.inference.speaker import EncoderClassifier

from src.api import PronunciationScorer, score
from tests.harness.augment import (
    add_additive_noise,
    add_synthetic_reverb,
    apply_codec_compression,
    apply_eq_tilt,
    mutate_phone_substitutions,
)

_SCORER = None
_ECAPA = None


def get_test_scorer():
    global _SCORER
    if _SCORER is None:
        _SCORER = PronunciationScorer()
    return _SCORER


def get_test_ecapa():
    global _ECAPA
    if _ECAPA is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _ECAPA = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
    return _ECAPA


def synthesize_wav(text: str) -> np.ndarray:
    """Synthesize 48 kHz float32 audio using pyopenjtalk."""
    x, sr = pyopenjtalk.tts(text)
    return (x / 32768.0).astype(np.float32)


class TestValidationHarness:
    """M4 Quality Gate Probe Suite."""

    def test_speaker_invariance_probe(self):
        """Probe 1: Score should be invariant to speaker pitch and rate variations."""
        scorer = get_test_scorer()
        ecapa = get_test_ecapa()

        ref_text = "こんにちは、本日はいい天気ですね"
        ref_wav = synthesize_wav(ref_text)
        ref_emb = ecapa.encode_batch(
            torch.from_numpy(ref_wav).float().unsqueeze(0).to(ecapa.device)
        ).squeeze()

        voices = {
            "native_base": ref_wav,
            "pitch_plus_2": librosa.effects.pitch_shift(ref_wav, sr=48000, n_steps=2.0),
            "pitch_minus_2": librosa.effects.pitch_shift(ref_wav, sr=48000, n_steps=-2.0),
            "pitch_plus_4": librosa.effects.pitch_shift(ref_wav, sr=48000, n_steps=4.0),
            "pitch_minus_4": librosa.effects.pitch_shift(ref_wav, sr=48000, n_steps=-4.0),
            "speed_0.9x": librosa.effects.time_stretch(ref_wav, rate=0.9),
            "speed_1.1x": librosa.effects.time_stretch(ref_wav, rate=1.1),
        }

        scores, ecapa_dists = [], []
        print("\n--- Speaker Invariance Probe ---")
        for name, wav in voices.items():
            res = scorer.score(ref_wav, ref_text, wav)
            emb = ecapa.encode_batch(
                torch.from_numpy(wav).float().unsqueeze(0).to(ecapa.device)
            ).squeeze()
            cos_sim = torch.nn.functional.cosine_similarity(ref_emb, emb, dim=0).item()
            ecapa_dist = 1.0 - cos_sim

            scores.append(res.score)
            ecapa_dists.append(ecapa_dist)
            print(f"  {name:15s} | Score: {res.score:6.2f} (GOP: {res.gop_score:6.2f}) | ECAPA Dist: {ecapa_dist:.4f}")

        # Correlation between score and speaker distance
        corr = float(np.corrcoef(scores, ecapa_dists)[0, 1])
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        print(f"  Summary: Mean={mean_score:.2f}, Std={std_score:.2f}, Pearson |r|={abs(corr):.4f}")

        # Assertions
        assert mean_score >= 65.0, f"Mean speaker invariance score too low: {mean_score}"
        assert abs(corr) < 0.50, f"Score correlates too strongly with speaker identity: r={corr}"

    def test_channel_invariance_probe(self):
        """Probe 2: Score should withstand realistic acoustic channel corruptions."""
        scorer = get_test_scorer()

        ref_text = "こんにちは、本日はいい天気ですね"
        ref_wav = synthesize_wav(ref_text)
        clean_score = scorer.score(ref_wav, ref_text, ref_wav).score

        channels = {
            "noise_20dB": add_additive_noise(ref_wav, snr_db=20.0),
            "reverb_0.2s": add_synthetic_reverb(ref_wav, rt60=0.20),
            "eq_telephone": apply_eq_tilt(ref_wav, tilt_type="telephone"),
            "eq_lowpass": apply_eq_tilt(ref_wav, tilt_type="lowpass"),
            "codec_ogg": apply_codec_compression(ref_wav, format="OGG"),
        }

        print("\n--- Channel Invariance Probe ---")
        for name, wav in channels.items():
            res = scorer.score(ref_wav, ref_text, wav)
            drop = clean_score - res.score
            print(f"  {name:15s} | Score: {res.score:6.2f} | Drop: {drop:5.2f} pts")
            # Realistic channels shouldn't collapse the score
            assert res.score >= 60.0, f"Channel corruption {name} caused excessive drop to {res.score}"

    def test_duration_sensitivity_probe(self):
        """Probe 3: Phonemic duration errors (long vowels, geminates) must cause a score drop."""
        scorer = get_test_scorer()

        print("\n--- Duration Sensitivity Probe ---")
        # Long vowel test: おばあさん (grandmother) vs おばさん (aunt)
        ref_lv = "おばあさん"
        ref_lv_wav = synthesize_wav(ref_lv)
        short_lv_wav = synthesize_wav("おばさん")

        res_lv_clean = scorer.score(ref_lv_wav, ref_lv, ref_lv_wav)
        res_lv_short = scorer.score(ref_lv_wav, ref_lv, short_lv_wav)
        drop_lv = res_lv_clean.score - res_lv_short.score
        print(f"  Long vowel correct: {res_lv_clean.score:.2f} -> Shortened: {res_lv_short.score:.2f} (Drop: {drop_lv:.2f} pts)")
        assert drop_lv >= 8.0, f"Long vowel shortening did not drop materially: drop={drop_lv:.2f}"

        # Geminate sokuon test: がっこう (school) vs がこう
        ref_gem = "がっこう"
        ref_gem_wav = synthesize_wav(ref_gem)
        short_gem_wav = synthesize_wav("がこう")

        res_gem_clean = scorer.score(ref_gem_wav, ref_gem, ref_gem_wav)
        res_gem_short = scorer.score(ref_gem_wav, ref_gem, short_gem_wav)
        drop_gem = res_gem_clean.score - res_gem_short.score
        print(f"  Geminate correct: {res_gem_clean.score:.2f} -> Removed: {res_gem_short.score:.2f} (Drop: {drop_gem:.2f} pts)")
        assert drop_gem >= 5.0, f"Geminate deletion did not drop materially: drop={drop_gem:.2f}"

    def test_monotonicity_probe(self):
        """Probe 4: Score must decrease monotonically with k phone substitutions."""
        scorer = get_test_scorer()

        base_text = "あさごはんをたべました"
        base_wav = synthesize_wav(base_text)

        mut_scores = []
        print("\n--- Monotonicity Probe ---")
        for k in range(4):
            mut_text = mutate_phone_substitutions(base_text, k=k) if k > 0 else base_text
            mut_wav = synthesize_wav(mut_text)
            res = scorer.score(base_wav, base_text, mut_wav)
            mut_scores.append(res.score)
            print(f"  k={k} mutations ({mut_text:15s}) => Score: {res.score:6.2f}")

        # Check monotonic non-increasing trend
        for i in range(len(mut_scores) - 1):
            assert mut_scores[i] >= mut_scores[i + 1] - 1.0, (
                f"Score not monotonic at k={i} ({mut_scores[i]}) vs k={i+1} ({mut_scores[i+1]})"
            )

    def test_discrimination_probe(self):
        """Probe 5: Scoring completely mismatched phrases must floor significantly."""
        scorer = get_test_scorer()

        diff_pairs = [
            ("こんにちは", "さようなら、またあした"),
            ("学校に行きます", "東京特許許可局"),
        ]

        print("\n--- Discrimination Probe ---")
        for ref_p, wrong_p in diff_pairs:
            r_wav = synthesize_wav(ref_p)
            w_wav = synthesize_wav(wrong_p)
            res_diff = scorer.score(r_wav, ref_p, w_wav)
            print(f"  Ref: \"{ref_p:10s}\" vs Wrong: \"{wrong_p:15s}\" => Score: {res_diff.score:6.2f} (DTW cost={res_diff.dtw_cost:.3f})")
            assert res_diff.score < 70.0, f"Discrimination score failed to floor: {res_diff.score}"


def run_all_probes():
    """Run full probe suite and print summary report."""
    suite = TestValidationHarness()
    suite.test_speaker_invariance_probe()
    suite.test_channel_invariance_probe()
    suite.test_duration_sensitivity_probe()
    suite.test_monotonicity_probe()
    suite.test_discrimination_probe()
    print("\nAll 5 M4 Validation Probes Passed Successfully!")


if __name__ == "__main__":
    run_all_probes()
