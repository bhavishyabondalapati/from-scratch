# String matching: KMP and Rabin-Karp

## Intuition
Naive search tries the pattern at every position and, after a mismatch, starts over one position
later, re-reading text it already looked at: O(n·m) in the worst case.

- **KMP (Knuth-Morris-Pratt):** never re-read the text. When a mismatch happens after matching
  some characters, those characters are known (they equal the start of the pattern). Precompute,
  for each prefix of the pattern, how much of it can be *reused* as the start of a new attempt.
- **Rabin-Karp:** compare fingerprints (hashes) instead of strings. A **rolling hash** updates
  the window's fingerprint in O(1) as it slides, and real comparison happens only when
  fingerprints agree.

## Core math
- Prefix function: `lps[i]` = longest proper prefix of `p[:i+1]` that is also its suffix
  (a "border"). On a mismatch with `k` characters matched, set `k = lps[k-1]` and retry.
- Polynomial hash: `H(s) = Σ s[i] · B^(m-1-i) mod M`.
- Rolling update: `H(next) = ((H − s[out]·B^(m-1)) · B + s[in]) mod M`.
- Collision probability per comparison is about 1/M for a random base; a large prime modulus
  (2⁶¹ − 1 here) makes false positives very rare, and verification makes them harmless.
- Period from a border: if the longest border of `s` has length `b`, `s` has period `n − b`.

## Complexity
| Algorithm | Preprocess | Search | Worst case | Extra space |
|---|---|---|---|---|
| Naive | – | O(n·m) | O(n·m) | O(1) |
| KMP | O(m) | O(n) | O(n + m) | O(m) |
| Rabin-Karp | O(m) | O(n) expected | O(n·m) (all windows collide) | O(1) |

## Common pitfalls
- KMP: confusing *proper* prefix (must be shorter than the string) — `lps[0]` is always 0.
- KMP: after a full match, forgetting `k = lps[k-1]` and missing overlapping occurrences.
- Rabin-Karp: skipping verification (wrong answers on collisions); tiny modulus; negative
  values after subtraction in languages where `%` can be negative.
- Using Python's built-in `hash()` as a rolling hash (it cannot be rolled).
- Empty pattern conventions (here: matches at every position, like `str.find`).

## When to use vs. alternatives
- Single pattern, guaranteed linear time → KMP (or the Z-algorithm, same power).
- Many patterns of the same length, 2D matching, plagiarism detection, longest repeated
  substring via binary search → Rabin-Karp's rolling hash.
- Many patterns of different lengths → Aho-Corasick (KMP on a trie).
- In practice: library `find` (Boyer-Moore-Horspool / two-way) is sublinear on typical text.

## Interview questions
1. **What does the LPS / failure table store?** For each pattern prefix, the length of its
   longest proper prefix that is also a suffix: where to resume after a mismatch.
2. **Why is KMP O(n + m)?** The text pointer never moves back; `k` increases at most n times in
   total, so it can decrease at most n times in total (amortized).
3. **How does a rolling hash update in O(1)?** Subtract the outgoing character's weighted term,
   multiply by the base, add the incoming character, all modulo M.
4. **What if two different strings hash equal?** Verify with a direct comparison; choose a large
   prime modulus (or double hashing) to make collisions rare.
5. **How do you check if a string is a repetition of a smaller block?** Compute the longest
   border `b`; it repeats iff `n % (n − b) == 0` and `b > 0`.
