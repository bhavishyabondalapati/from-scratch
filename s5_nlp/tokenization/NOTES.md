# Tokenization: whitespace/regex, BPE from scratch, WordPiece

## Intuition
Models work on numbers, so text must be cut into pieces (tokens) that each get an id. The choice
of pieces is a trade-off:
- **Words** are meaningful but there are millions of them, and any new word is "unknown".
- **Characters/bytes** never fail, but sequences get very long and each token means little.
- **Subwords** (BPE, WordPiece) sit in between: frequent words become single tokens, rare words
  are spelled out from smaller pieces (`" foxes"` → `" fox" + "e" + "s"` after a few merges).

**Byte-pair encoding** starts from single bytes and repeatedly glues together the most frequent
adjacent pair, adding it to the vocabulary. **WordPiece** (BERT) is similar but picks the pair whose
parts appear together *more than chance*, and encodes greedily by longest match.

## Core math / algorithms
- **BPE training:** vocab = 256 bytes; repeat `V − 256` times: count adjacent pairs (weighted by
  word frequency), merge the most frequent pair `(a, b) → new id`. Pre-tokenization (a regex that
  splits words, numbers, punctuation, attaching a leading space) keeps merges inside words.
- **BPE encoding:** within each pre-token, repeatedly apply the *earliest-learned* merge present.
- Byte level ⇒ every string is encodable; `decode(encode(s)) == s` (tested on random Unicode).
- **WordPiece score:** `count(ab) / (count(a)·count(b))`; encoding = greedy longest-match-first,
  continuation pieces prefixed `##`; if any position cannot be matched the word is `[UNK]`.
- Sennrich et al.'s toy corpus `{low:5, lower:2, newest:6, widest:3}` gives first merges
  `e s`, `es t`, `est </w>`, `l o` (reproduced in the tests).
- On the test corpus: 44 merges cut 4,020 bytes to 1,652 tokens; 144 merges to 871.

## Complexity
Naive BPE training: O(merges × corpus size). Counting each distinct pre-token once (as here) makes
it O(merges × unique words × word length). Encoding a word of length L with this simple loop is
O(L²); production tokenizers use priority queues / tries and are written in Rust or C++.

## Common pitfalls
- Different tokenizers for training and inference (ids mean different things).
- Whitespace handling: GPT-style tokens include the leading space (`" the"` ≠ `"the"`).
- Counting characters instead of tokens for context limits and pricing.
- Unicode: splitting a multi-byte character mid-way; byte-level BPE handles it, char-level may not.
- Numbers and code tokenize poorly (digits split unevenly), which hurts arithmetic.

## When to use vs. alternatives
- Generative LMs: byte-level BPE (GPT-2/3/4, Llama 3). BERT-family: WordPiece. Many multilingual
  models: SentencePiece (BPE or unigram LM over raw text, treats space as a symbol).
- Classic NLP pipelines / bag-of-words: regex word tokenization + normalisation.

## Interview questions
1. **Why subword tokenization?** Bounded vocabulary, no out-of-vocabulary words, and short
   sequences for common text.
2. **How does BPE training work?** Start from characters/bytes and repeatedly merge the most
   frequent adjacent pair into a new token.
3. **BPE vs WordPiece?** BPE merges by raw frequency; WordPiece by a likelihood ratio; WordPiece
   encodes greedily by longest match and marks continuations with `##`.
4. **Why byte-level BPE?** The 256-byte base alphabet can represent any Unicode string, so there
   is never an unknown token.
5. **How does vocabulary size trade off?** Bigger vocab → shorter sequences but a larger embedding
   matrix and rarer, less-trained tokens.
