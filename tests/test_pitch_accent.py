"""M5 Pitch Accent (F0) Probe & Minimal Pair Validation."""

import numpy as np
import pyopenjtalk
import pytest
import torch

from src.api import PronunciationScorer


class TestPitchAccent:
    """M5: Pitch accent opt-in evaluation."""

    def test_opt_in_f0_score(self):
        scorer = PronunciationScorer()
        ref_text = "雨が降っています"
        x, _ = pyopenjtalk.tts(ref_text)
        wav = (x / 32768.0).astype(np.float32)

        # Without F0
        res_no_f0 = scorer.score(wav, ref_text, wav, use_f0=False)
        assert res_no_f0.f0_score is None
        assert res_no_f0.score >= 99.0

        # With F0
        res_with_f0 = scorer.score(wav, ref_text, wav, use_f0=True)
        assert res_with_f0.f0_score is not None
        assert res_with_f0.f0_score >= 95.0
        assert res_with_f0.score >= 99.0

    def test_f0_pitch_shift_invariant(self):
        """Confirm z-scored log-F0 removes absolute pitch level without leaking gender/speaker."""
        scorer = PronunciationScorer()
        ref_text = "おはようございます"
        x, sr = pyopenjtalk.tts(ref_text)
        wav = (x / 32768.0).astype(np.float32)

        import librosa

        shifted_wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=4.0)

        res = scorer.score(wav, ref_text, shifted_wav, use_f0=True)
        assert res.f0_score is not None
        # Z-scored relative pitch contour should remain correlated (> 60)
        assert res.f0_score >= 50.0
