import random

import pytest

from .impl import (
    BPETokenizer,
    regex_tokenize,
    sennrich_bpe,
    train_wordpiece,
    whitespace_tokenize,
    wordpiece_tokenize,
)

CORPUS = (
    "The quick brown fox jumps over the lazy dog. The dog sleeps; the fox runs! "
    "Tokenizers turn text into tokens, and tokens into numbers. "
) * 30


def test_simple_splitters():
    assert whitespace_tokenize("  hello   world \n") == ["hello", "world"]
    assert regex_tokenize("Don't stop, believing!") == ["don't", "stop", ",", "believing", "!"]
    assert regex_tokenize("GPT-2 costs $5.") == ["gpt", "-", "2", "costs", "$", "5", "."]


def test_sennrich_paper_example():
    # The toy corpus from Sennrich et al. (2016), Figure 1 / Section 3.2.
    merges = sennrich_bpe({"low": 5, "lower": 2, "newest": 6, "widest": 3}, 4)
    assert merges == [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o")]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer().train(CORPUS, vocab_size=300)


def test_bpe_round_trips_any_text(tok):
    rnd = random.Random(0)
    alphabet = "abcxyz ,.!?\n\tÄéß日本語🙂"
    for _ in range(300):
        s = "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, 40)))
        assert tok.decode(tok.encode(s)) == s  # byte-level: nothing is ever unknown
    assert tok.decode(tok.encode(CORPUS)) == CORPUS


def test_bpe_compresses_and_learns_frequent_words(tok):
    n_bytes = len(CORPUS.encode("utf-8"))
    assert len(tok.encode(CORPUS)) < 0.45 * n_bytes  # 44 merges already remove >55% of tokens
    assert len(tok.merges) == 300 - 256
    assert " the" in tok.tokens(" the fox") or "the" in tok.tokens("the")
    # Each merged token is exactly the concatenation of the pair that created it.
    for (a, b), new in tok.merges.items():
        assert tok.vocab[new] == tok.vocab[a] + tok.vocab[b]


def test_more_merges_means_fewer_tokens():
    lengths = [len(BPETokenizer().train(CORPUS, v).encode(CORPUS)) for v in (256, 270, 300, 400)]
    assert lengths[0] == len(CORPUS.encode("utf-8"))  # no merges: one token per byte
    assert all(a >= b for a, b in zip(lengths, lengths[1:]))


def test_encoding_unseen_words_falls_back_to_pieces(tok):
    toks = tok.tokens(" foxes jumped")
    assert "".join(toks) == " foxes jumped" and len(toks) < len(" foxes jumped")
    with pytest.raises(ValueError):
        BPETokenizer().train("abc", 100)


def test_wordpiece_greedy_longest_match():
    vocab = {"un", "##aff", "##able", "##a", "##ff", "play", "##ing", "##s", "a"}
    assert wordpiece_tokenize("unaffable", vocab) == ["un", "##aff", "##able"]
    assert wordpiece_tokenize("playing", vocab) == ["play", "##ing"]
    assert wordpiece_tokenize("plays", vocab) == ["play", "##s"]
    assert wordpiece_tokenize("xyz", vocab) == ["[UNK]"]  # whole word becomes unknown


def test_wordpiece_training_prefers_informative_pairs():
    counts = {"hugging": 10, "hug": 8, "pug": 3, "pun": 5, "bun": 4, "hugs": 6}
    vocab = train_wordpiece(counts, 25)
    assert {"h", "##u", "##g"} <= vocab  # starts from characters
    # Every training word can now be tokenized without [UNK], and pieces rebuild the word.
    for w in counts:
        pieces = wordpiece_tokenize(w, vocab)
        assert "[UNK]" not in pieces
        assert pieces[0] + "".join(p[2:] for p in pieces[1:]) == w
    assert len(wordpiece_tokenize("hugging", vocab)) < len("hugging")
