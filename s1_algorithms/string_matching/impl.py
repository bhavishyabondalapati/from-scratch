"""String matching: KMP and Rabin-Karp. Both find a pattern in a text in O(n + m)."""

from __future__ import annotations


def prefix_function(pattern: str) -> list[int]:
    """KMP failure table (a.k.a. LPS array).

    lps[i] = length of the longest PROPER prefix of pattern[: i + 1] that is
    also a suffix of it. Example: "ababaca" -> [0, 0, 1, 2, 3, 0, 1].

    Meaning for searching: if we matched i + 1 characters and then failed, the
    last lps[i] characters we matched are also the first lps[i] characters of
    the pattern, so we can resume from there without re-reading the text.
    """
    lps = [0] * len(pattern)
    k = 0  # length of the current matched prefix
    for i in range(1, len(pattern)):
        while k > 0 and pattern[i] != pattern[k]:
            k = lps[k - 1]  # fall back to the next shorter prefix-suffix
        if pattern[i] == pattern[k]:
            k += 1
        lps[i] = k
    return lps


def kmp_search(text: str, pattern: str) -> list[int]:
    """All start indices where pattern occurs in text (overlaps included). O(n + m)."""
    if not pattern:
        return list(range(len(text) + 1))
    lps = prefix_function(pattern)
    matches: list[int] = []
    k = 0  # number of pattern characters currently matched
    for i, ch in enumerate(text):  # the text pointer i never moves backwards
        while k > 0 and ch != pattern[k]:
            k = lps[k - 1]
        if ch == pattern[k]:
            k += 1
        if k == len(pattern):
            matches.append(i - k + 1)
            k = lps[k - 1]  # keep going to find overlapping matches
    return matches


def rabin_karp_search(
    text: str, pattern: str, base: int = 257, mod: int = (1 << 61) - 1
) -> list[int]:
    """All start indices of pattern in text using a rolling hash.

    hash(s) = (s[0]*base^(m-1) + s[1]*base^(m-2) + ... + s[m-1]) mod `mod`.
    Sliding the window one step is O(1): remove the leading character's term,
    multiply by base, add the new character. Equal hashes are VERIFIED with a
    real comparison, so collisions cost time but never correctness.
    Expected O(n + m); worst case O(n * m) if hashes keep colliding.
    """
    n, m = len(text), len(pattern)
    if m == 0:
        return list(range(n + 1))
    if m > n:
        return []
    high = pow(base, m - 1, mod)  # weight of the window's leading character
    target = window = 0
    for i in range(m):
        target = (target * base + ord(pattern[i])) % mod
        window = (window * base + ord(text[i])) % mod
    matches: list[int] = []
    for start in range(n - m + 1):
        if window == target and text[start : start + m] == pattern:
            matches.append(start)
        if start + m < n:
            window = (window - ord(text[start]) * high) % mod  # drop the old character
            window = (window * base + ord(text[start + m])) % mod  # bring in the new one
    return matches


def longest_prefix_also_suffix(s: str) -> str:
    """Longest proper prefix of s that is also a suffix ("happy prefix")."""
    return s[: prefix_function(s)[-1]] if s else ""


def shortest_repeating_unit(s: str) -> str:
    """Smallest block that repeats to form s, or s itself if there is none.

    If the longest border has length b, then s has period p = n - b; s is a
    whole number of copies of s[:p] exactly when p divides n.
    """
    n = len(s)
    if n == 0:
        return ""
    period = n - prefix_function(s)[-1]
    return s[:period] if n % period == 0 else s


def longest_repeated_substring(s: str) -> str:
    """Longest substring that occurs at least twice (overlaps allowed).

    Binary search on the length L (if a repeat of length L exists, one of
    length L - 1 does too), checking each L with rolling hashes in O(n).
    """
    base, mod = 257, (1 << 61) - 1

    def find(length: int) -> int:
        """Start index of some substring of this length seen twice, or -1."""
        if length == 0:
            return 0
        high = pow(base, length - 1, mod)
        h = 0
        for ch in s[:length]:
            h = (h * base + ord(ch)) % mod
        seen: dict[int, list[int]] = {h: [0]}
        for start in range(1, len(s) - length + 1):
            h = (h - ord(s[start - 1]) * high) % mod
            h = (h * base + ord(s[start + length - 1])) % mod
            for earlier in seen.get(h, []):
                if s[earlier : earlier + length] == s[start : start + length]:  # verify
                    return start
            seen.setdefault(h, []).append(start)
        return -1

    lo, hi, best_start = 0, max(len(s) - 1, 0), 0
    while lo < hi:
        mid = (lo + hi + 1) // 2  # round up because we set lo = mid
        start = find(mid)
        if start != -1:
            lo, best_start = mid, start
        else:
            hi = mid - 1
    return s[best_start : best_start + lo]
