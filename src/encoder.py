"""Acoustic Model & Feature Extractor.

Wraps facebook/wav2vec2-xlsr-53-espeak-cv-ft:
- Runs in fp16 on CUDA (or fp32 on CPU)
- Extracts CTC log-posteriors (PPG) at ~50 Hz
- Optionally extracts layer-9 hidden states with per-utterance CMVN and per-frame L2 norm
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCTC, AutoProcessor


@dataclass
class EncoderOutput:
    """Output dataclass from AcousticEncoder."""

    logits: torch.Tensor  # (1, T, V)
    log_probs: torch.Tensor  # (1, T, V) log-posteriors (PPG)
    hidden_states: Optional[torch.Tensor] = None  # (1, T, D) normalized layer-9 features


class AcousticEncoder:
    """wav2vec2-xlsr-53-espeak acoustic encoder."""

    def __init__(
        self,
        model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if dtype is None:
            self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        else:
            self.dtype = dtype

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCTC.from_pretrained(model_id)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        self.vocab: Dict[str, int] = self.processor.tokenizer.get_vocab()
        self.inv_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.blank_id: int = self.vocab.get("<pad>", 0)

    @torch.no_grad()
    def extract(
        self,
        wav: Union[torch.Tensor, np.ndarray],
        return_hidden_states: bool = False,
        layer_idx: int = 9,
    ) -> EncoderOutput:
        """Extract frame-level CTC log-posteriors (PPG) and optional normalized hidden states.

        Args:
            wav: 1D or 2D audio tensor/array sampled at 16 kHz.
            return_hidden_states: If True, extract normalized layer-9 hidden states.
            layer_idx: Layer index for hidden state extraction (default: 9).

        Returns:
            EncoderOutput containing logits, log_probs (PPG), and optional hidden_states.
        """
        if isinstance(wav, np.ndarray):
            wav_t = torch.from_numpy(wav).float()
        elif isinstance(wav, torch.Tensor):
            wav_t = wav.float()
        else:
            raise TypeError(f"Unsupported audio type: {type(wav)}")

        if wav_t.ndim == 1:
            wav_t = wav_t.unsqueeze(0)  # (1, T)

        # Move to model device & dtype
        wav_t = wav_t.to(device=self.device, dtype=self.dtype)

        outputs = self.model(wav_t, output_hidden_states=return_hidden_states)
        logits = outputs.logits  # (1, T_frames, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)

        norm_hidden: Optional[torch.Tensor] = None
        if return_hidden_states and outputs.hidden_states is not None:
            raw_hs = outputs.hidden_states[layer_idx]  # (1, T_frames, hidden_dim)
            # Per-utterance CMVN
            mean = raw_hs.mean(dim=1, keepdim=True)
            std = raw_hs.std(dim=1, keepdim=True) + 1e-7
            hs_cmvn = (raw_hs - mean) / std
            # Per-frame L2 normalization
            l2_norm = torch.norm(hs_cmvn, p=2, dim=-1, keepdim=True) + 1e-7
            norm_hidden = hs_cmvn / l2_norm

        return EncoderOutput(
            logits=logits,
            log_probs=log_probs,
            hidden_states=norm_hidden,
        )

    def decode_greedy(self, log_probs: torch.Tensor) -> str:
        """Greedy CTC decode of log_probs or logits tensor."""
        if log_probs.ndim == 3:
            pred_ids = torch.argmax(log_probs, dim=-1)[0]
        else:
            pred_ids = torch.argmax(log_probs, dim=-1)
        return self.processor.decode(pred_ids)


_ENCODER_SINGLETON: Optional[AcousticEncoder] = None


def get_encoder(
    model_id: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
    device: Optional[str] = None,
) -> AcousticEncoder:
    """Get or create singleton AcousticEncoder instance."""
    global _ENCODER_SINGLETON
    if _ENCODER_SINGLETON is None:
        _ENCODER_SINGLETON = AcousticEncoder(model_id=model_id, device=device)
    return _ENCODER_SINGLETON
