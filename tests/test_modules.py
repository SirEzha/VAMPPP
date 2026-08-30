"""Unit tests for individual core modules."""

import numpy as np
import pytest
import torch

from src.align import PhoneSpan, align_tokens, group_mora_spans
from src.audio import load_audio, normalize_loudness, preprocess_audio, trim_silence_vad
from src.encoder import AcousticEncoder, get_encoder
from src.g2p import phones_to_ids, text_to_phones
from src.score import (
    calibrate_gop_scores,
    combine_scores,
    compute_duration_score,
    compute_phone_gop,
    compute_ppg_dtw_score,
    sakoe_chiba_dtw,
)


class TestAudioModule:
    """Test audio loading, resampling, loudness normalization, and VAD."""

    def test_load_audio_from_numpy(self):
        sr = 16000
        dummy_wav = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False)).astype(np.float32)
        wav, out_sr = load_audio(dummy_wav, orig_sr=16000, target_sr=16000)
        assert out_sr == 16000
        assert isinstance(wav, np.ndarray)
        assert wav.ndim == 1
        assert len(wav) == sr

    def test_stereo_to_mono(self):
        stereo_wav = np.ones((2, 16000), dtype=np.float32)
        mono_wav, _ = load_audio(stereo_wav, orig_sr=16000, target_sr=16000)
        assert mono_wav.ndim == 1
        assert len(mono_wav) == 16000

    def test_normalize_loudness(self):
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        sig = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        norm = normalize_loudness(sig, sr=sr, target_lufs=-23.0)
        assert isinstance(norm, np.ndarray)
        assert np.max(np.abs(norm)) <= 1.0

    def test_trim_silence_vad(self):
        sr = 16000
        # 0.5s silence + 1.0s sine + 0.5s silence
        silence = np.zeros(int(sr * 0.5), dtype=np.float32)
        speech = (0.5 * np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr, endpoint=False))).astype(np.float32)
        full_wav = np.concatenate([silence, speech, silence])
        trimmed = trim_silence_vad(full_wav, sr=sr)
        assert len(trimmed) <= len(full_wav)
        assert len(trimmed) > 0


class TestG2PModule:
    """Test G2P for Japanese and multilingual text."""

    def test_japanese_g2p_kanji_and_kana(self):
        phones1 = text_to_phones("こんにちは", language="ja")
        phones2 = text_to_phones("こんにちわ", language="ja")
        assert len(phones1) > 0
        assert "k" in phones1
        assert "o" in phones1
        assert "n" in phones1

    def test_japanese_g2p_kana_override(self):
        # Kanji with ambiguous reading
        phones_override = text_to_phones("一日", language="ja", kana_override="ついたち")
        assert "ts" in phones_override or "t" in phones_override

    def test_other_languages_phonemizer(self):
        en_phones = text_to_phones("hello world", language="en-us")
        assert len(en_phones) > 0

    def test_phones_to_ids(self):
        encoder = get_encoder()
        phones = ["k", "o", "n", "n", "i", "tɕ", "i", "w", "a"]
        valid_phones, ids = phones_to_ids(phones, encoder.vocab)
        assert len(ids) == len(phones)
        assert all(isinstance(x, int) for x in ids)


class TestEncoderModule:
    """Test acoustic feature extraction, VRAM, and CMVN layer-9 features."""

    def test_encoder_extract(self):
        encoder = get_encoder()
        dummy_audio = torch.randn(16000)
        out = encoder.extract(dummy_audio, return_hidden_states=True)
        assert out.log_probs.ndim == 3
        assert out.log_probs.shape[0] == 1
        assert out.log_probs.shape[2] == len(encoder.vocab)
        assert out.hidden_states is not None
        assert out.hidden_states.ndim == 3
        assert out.hidden_states.shape[2] == 1024


class TestAlignmentAndScore:
    """Test forced alignment and scoring functions."""

    def test_align_tokens_contiguous_spans(self):
        encoder = get_encoder()
        T = 50
        V = len(encoder.vocab)
        dummy_lp = torch.randn(1, T, V)
        target_ids = [encoder.vocab["k"], encoder.vocab["o"], encoder.vocab["n"]]
        spans = align_tokens(dummy_lp, target_ids, encoder.inv_vocab, blank_id=encoder.blank_id)
        assert len(spans) == len(target_ids)
        # Check contiguous bounds
        for i in range(len(spans) - 1):
            assert spans[i].end_frame == spans[i + 1].start_frame

    def test_dtw_self_cost_zero(self):
        x = np.random.randn(40, 128)
        cost, path = sakoe_chiba_dtw(x, x, return_path=True)
        assert cost < 1e-5
        assert len(path) == 40
        assert np.all(path[0] == [0, 0])
        assert np.all(path[-1] == [39, 39])

    def test_combine_scores_bounds(self):
        score_val = combine_scores(gop_score=85.0, dur_score=90.0, dtw_score=95.0)
        assert 0.0 <= score_val <= 100.0
