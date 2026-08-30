"""Grapheme-to-Phoneme (G2P) conversion.

Supports:
- Japanese: pyopenjtalk + static OpenJTalk->eSpeak-IPA mapping table
  (with kana/furigana override for ambiguous kanji readings)
- Other languages: phonemizer (eSpeak backend)
- Caching per line
"""

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import pyopenjtalk
from phonemizer import phonemize

# Static OpenJTalk phoneme to eSpeak-IPA mapping table
OJT_TO_ESPEAK_IPA: Dict[str, str] = {
    # Vowels (standard and devoiced)
    "a": "a",
    "i": "i",
    "u": "ɯ",
    "e": "e",
    "o": "o",
    "A": "a",
    "I": "i",
    "U": "ɯ",
    "E": "e",
    "O": "o",
    # Consonants
    "k": "k",
    "ky": "kʲ",
    "g": "ɡ",
    "gy": "ɡʲ",
    "s": "s",
    "sh": "ɕ",
    "z": "z",
    "j": "dʑ",
    "t": "t",
    "ch": "tɕ",
    "ts": "ts",
    "d": "d",
    "dy": "dʲ",
    "n": "n",
    "ny": "nʲ",
    "h": "h",
    "hy": "ç",
    "f": "ɸ",
    "b": "b",
    "by": "bʲ",
    "p": "p",
    "py": "pʲ",
    "m": "m",
    "my": "mʲ",
    "y": "j",
    "r": "ɾ",
    "ry": "rʲ",
    "w": "w",
    "v": "v",
    # Moraic nasal (ん)
    "N": "n",
}

# Consonant gemination mapping for 'cl' (sokuon っ)
SOKUON_CONSONANT_MAP: Dict[str, str] = {
    "k": "k",
    "ky": "kʲ",
    "g": "ɡ",
    "gy": "ɡʲ",
    "s": "s",
    "sh": "ɕ",
    "t": "t",
    "ch": "tɕ",
    "ts": "ts",
    "d": "d",
    "p": "p",
    "py": "pʲ",
    "b": "b",
    "by": "bʲ",
    "f": "ɸ",
}


@lru_cache(maxsize=1024)
def _cached_japanese_g2p(text: str, kana_override: Optional[str] = None) -> Tuple[str, ...]:
    """Cached internal Japanese G2P conversion."""
    source_text = kana_override if kana_override is not None else text
    raw_ojt = pyopenjtalk.g2p(source_text)
    tokens = raw_ojt.strip().split()

    ipa_tokens: List[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in ("pau", "sil", "", " "):
            i += 1
            continue

        if t == "cl":
            # Geminate next consonant
            if i + 1 < len(tokens):
                next_t = tokens[i + 1]
                gem_sym = SOKUON_CONSONANT_MAP.get(
                    next_t, OJT_TO_ESPEAK_IPA.get(next_t, next_t)
                )
                ipa_tokens.append(gem_sym)
            i += 1
            continue

        ipa_sym = OJT_TO_ESPEAK_IPA.get(t, t)
        ipa_tokens.append(ipa_sym)
        i += 1

    return tuple(ipa_tokens)


@lru_cache(maxsize=1024)
def _cached_other_g2p(text: str, language: str = "en-us") -> Tuple[str, ...]:
    """Cached internal non-Japanese G2P conversion using phonemizer."""
    phonemes_str = phonemize(
        text,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
    )
    # Split into characters / IPA tokens
    tokens = [c for c in phonemes_str if c not in (" ", "\n", "\t")]
    return tuple(tokens)


def text_to_phones(
    text: str,
    language: str = "ja",
    kana_override: Optional[str] = None,
) -> List[str]:
    """Convert text to a sequence of IPA phone strings.

    Args:
        text: Input text (e.g. Japanese kanji/kana, English, etc.).
        language: Language code ('ja', 'en-us', 'sm', etc.).
        kana_override: Optional explicit kana/furigana to bypass kanji ambiguity.

    Returns:
        List of IPA phoneme strings.
    """
    lang_lower = language.lower().strip()
    if lang_lower in ("ja", "jp", "japanese"):
        return list(_cached_japanese_g2p(text, kana_override))
    else:
        return list(_cached_other_g2p(text, language))


def phones_to_ids(
    phones: List[str],
    vocab: Dict[str, int],
) -> Tuple[List[str], List[int]]:
    """Map phone strings to vocabulary token IDs, filtering unknown symbols.

    Args:
        phones: List of IPA phone strings.
        vocab: Dictionary mapping token strings to integer IDs.

    Returns:
        Tuple of (valid_phone_strings, token_ids).
    """
    valid_phones: List[str] = []
    token_ids: List[int] = []

    for p in phones:
        if p in vocab:
            valid_phones.append(p)
            token_ids.append(vocab[p])
        else:
            # Try decomposing multi-char phone
            for c in p:
                if c in vocab:
                    valid_phones.append(c)
                    token_ids.append(vocab[c])

    return valid_phones, token_ids
