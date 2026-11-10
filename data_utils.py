"""Small dataset downloads, cached under data/ (which is gitignored).

Only the standard library is used. Each file is downloaded once; later calls read the cache.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download(url: str, filename: str) -> Path:
    path = DATA_DIR / filename
    if not path.exists():
        DATA_DIR.mkdir(exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (fixed, trusted URL)
        tmp.rename(path)
    return path


def tiny_shakespeare() -> str:
    """~1.1 MB of Shakespeare plays (Karpathy's char-rnn dataset)."""
    return download(TINY_SHAKESPEARE_URL, "tinyshakespeare.txt").read_text(encoding="utf-8")
