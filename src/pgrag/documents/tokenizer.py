"""Offline tokenizer for the production embedder (bge-small-en-v1.5).

The corpus build must run with no servers up, but chunking has to respect the
embedder's hard 512-token window — a character budget cannot guarantee that,
because token density varies wildly in game text (~2.4 c/t for level tables vs
~4.9 c/t for prose). This loads the exact BERT WordPiece tokenizer the embed
server uses (vendored at ``pgrag/resources/bge_tokenizer.json``) so chunk budgets are
measured in tokens, deterministically and offline.

Reproduction check: this tokenizer returns the same counts llama-server reports
for the same input (``input (N tokens) is larger than the max context size``).
"""

from __future__ import annotations

import importlib.resources as resources
import logging

_logger = logging.getLogger(__name__)

try:
    from tokenizers import Tokenizer
except ImportError:  # pragma: no cover - dependency absent
    Tokenizer = None  # type: ignore[assignment]

_tokenizer: Tokenizer | False | None = None  # None = not loaded, False = unavailable


def load_tokenizer() -> Tokenizer | None:
    """Load the bge-small tokenizer from the vendored asset, or None if unavailable."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer or None
    if Tokenizer is None:
        _tokenizer = False
        return None
    asset = resources.files("pgrag") / "resources" / "bge_tokenizer.json"
    if not asset.is_file():
        _logger.warning("bge tokenizer asset missing at %s; token chunking unavailable", asset)
        _tokenizer = False
        return None
    _tokenizer = Tokenizer.from_file(str(asset))
    return _tokenizer


def token_count(text: str) -> int | None:
    """WordPiece token count for ``text`` (no special tokens), or None if no tokenizer."""
    tok = load_tokenizer()
    if tok is None:
        return None
    return len(tok.encode(text, add_special_tokens=False).ids)
