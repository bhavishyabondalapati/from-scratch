"""Tokenization: whitespace/regex splitting, byte-level BPE trained from scratch, and WordPiece."""

from __future__ import annotations

import re
from collections import Counter

# --- Simple splitters ---------------------------------------------------------------


def whitespace_tokenize(text: str) -> list[str]:
    return text.split()


# Words (letters/digits/apostrophes inside words), or any single non-space symbol.
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]")


def regex_tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Split off punctuation: "Don't stop!" -> ["don't", "stop", "!"]."""
    return WORD_RE.findall(text.lower() if lowercase else text)


# GPT-2 style pre-tokenizer: BPE merges never cross these chunk boundaries, so
# "dog." and "dog!" share the token "dog", and spaces attach to the word after them.
PRETOKEN_RE = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


# --- Byte-level BPE -------------------------------------------------------------------


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` (left to right) with new_id."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-pair encoding over UTF-8 bytes (like GPT-2/GPT-4 tokenizers).

    Start from the 256 possible bytes, so ANY string can be encoded (no unknown
    tokens). Training repeatedly finds the most frequent adjacent pair of tokens
    in the corpus and gives it a new id. Encoding replays those merges in the
    order they were learned.
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) -> new id, in learning order
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> BPETokenizer:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 (the byte alphabet)")
        # Count each pre-token once with its frequency instead of scanning the raw text.
        words = Counter(PRETOKEN_RE.findall(text))
        seqs = {w: list(w.encode("utf-8")) for w in words}
        for new_id in range(256, vocab_size):
            pairs: Counter[tuple[int, int]] = Counter()
            for w, ids in seqs.items():
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += words[w]
            if not pairs:
                break  # every word is a single token already
            # Most frequent pair; ties broken by the smaller pair for determinism.
            best = max(pairs, key=lambda p: (pairs[p], (-p[0], -p[1])))
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            seqs = {w: _merge(ids, best, new_id) for w, ids in seqs.items()}
        return self

    def _encode_chunk(self, chunk: str) -> list[int]:
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the EARLIEST-learned merge available in this chunk.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for chunk in PRETOKEN_RE.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[str]:
        return [self.vocab[i].decode("utf-8", errors="replace") for i in self.encode(text)]


def sennrich_bpe(word_counts: dict[str, int], n_merges: int) -> list[tuple[str, str]]:
    """The original character-level BPE (Sennrich et al., 2016), with an end-of-word marker.
    Returns the learned merges; kept separate because it is the version in the paper."""
    vocab = {tuple(w) + ("</w>",): c for w, c in word_counts.items()}
    merges = []
    for _ in range(n_merges):
        pairs: Counter[tuple[str, str]] = Counter()
        for symbols, c in vocab.items():
            for pair in zip(symbols, symbols[1:]):
                pairs[pair] += c
        if not pairs:
            break
        best = max(pairs, key=pairs.get)  # ties: first pair encountered, as in the paper's code
        merges.append(best)
        new_vocab = {}
        for symbols, c in vocab.items():
            out, i = [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                    out.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            new_vocab[tuple(out)] = c
        vocab = new_vocab
    return merges


# --- WordPiece (BERT) ---------------------------------------------------------------


def wordpiece_tokenize(
    word: str, vocab: set[str], unk: str = "[UNK]", max_chars: int = 100
) -> list[str]:
    """Greedy longest-match-first, as in BERT. Pieces after the first are marked "##".

    "unaffable" with vocab {un, ##aff, ##able} -> ["un", "##aff", "##able"].
    If some position cannot be matched at all, the WHOLE word becomes [UNK].
    """
    if len(word) > max_chars:
        return [unk]
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        piece = None
        while start < end:
            candidate = word[start:end] if start == 0 else "##" + word[start:end]
            if candidate in vocab:
                piece = candidate
                break
            end -= 1
        if piece is None:
            return [unk]
        pieces.append(piece)
        start = end
    return pieces


def train_wordpiece(word_counts: dict[str, int], vocab_size: int) -> set[str]:
    """Learn a WordPiece vocabulary. Like BPE, but merge the pair with the highest

        score(a, b) = count(ab) / (count(a) * count(b))

    i.e. pairs that occur together MORE than their parts' frequencies would
    predict (a likelihood gain), rather than simply the most frequent pair.
    """
    splits = {w: [w[0]] + ["##" + ch for ch in w[1:]] for w in word_counts}
    vocab = {p for pieces in splits.values() for p in pieces}
    while len(vocab) < vocab_size:
        pair_counts: Counter[tuple[str, str]] = Counter()
        piece_counts: Counter[str] = Counter()
        for w, pieces in splits.items():
            c = word_counts[w]
            for p in pieces:
                piece_counts[p] += c
            for pair in zip(pieces, pieces[1:]):
                pair_counts[pair] += c
        if not pair_counts:
            break
        best = max(
            pair_counts,
            key=lambda p: (pair_counts[p] / (piece_counts[p[0]] * piece_counts[p[1]]), p),
        )
        a, b = best
        merged = a + b[2:]  # "##" is dropped from the right-hand piece
        vocab.add(merged)
        for w, pieces in splits.items():
            out, i = [], 0
            while i < len(pieces):
                if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == best:
                    out.append(merged)
                    i += 2
                else:
                    out.append(pieces[i])
                    i += 1
            splits[w] = out
    return vocab
