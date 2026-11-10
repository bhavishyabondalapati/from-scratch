import random
import re

import pytest

from .impl import (
    kmp_search,
    longest_prefix_also_suffix,
    longest_repeated_substring,
    prefix_function,
    rabin_karp_search,
    shortest_repeating_unit,
)


def all_occurrences(text, pattern):
    """Reference: overlapping matches via a regex lookahead."""
    return [m.start() for m in re.finditer(f"(?={re.escape(pattern)})", text)]


def rand_str(rnd, alphabet, max_len):
    return "".join(rnd.choice(alphabet) for _ in range(rnd.randint(0, max_len)))


@pytest.mark.parametrize("search", [kmp_search, rabin_karp_search])
def test_search_matches_reference(search):
    rnd = random.Random(0)
    for _ in range(600):
        text = rand_str(rnd, "ab", 30)
        pattern = rand_str(rnd, "ab", 4)
        assert search(text, pattern) == all_occurrences(text, pattern)
    assert search("aaaa", "aa") == [0, 1, 2]  # overlapping matches
    assert search("abc", "abcd") == [] and search("", "") == [0]
    assert search("héllo wörld ✓ héllo", "héllo") == [0, 14]  # non-ASCII text


def test_rabin_karp_is_correct_even_when_every_hash_collides():
    rnd = random.Random(1)
    for _ in range(200):
        text, pattern = rand_str(rnd, "abc", 25), rand_str(rnd, "abc", 3)
        # mod=1 makes every hash 0: only the verification step keeps it correct.
        assert rabin_karp_search(text, pattern, mod=1) == all_occurrences(text, pattern)
        assert rabin_karp_search(text, pattern, base=3, mod=7) == all_occurrences(text, pattern)


def test_prefix_function_against_definition():
    assert prefix_function("ababaca") == [0, 0, 1, 2, 3, 0, 1]
    assert prefix_function("aaaa") == [0, 1, 2, 3]
    rnd = random.Random(2)
    for _ in range(300):
        s = rand_str(rnd, "ab", 14)
        expected = [
            max(k for k in range(i + 1) if s[:k] == s[i + 1 - k : i + 1]) for i in range(len(s))
        ]
        assert prefix_function(s) == expected


def test_kmp_is_linear_on_a_worst_case_for_naive_search():
    # Naive search does ~n*m character comparisons here; KMP must not.
    text, pattern = "a" * 200_000 + "b", "a" * 1000 + "b"
    assert kmp_search(text, pattern) == [199_000]
    assert rabin_karp_search(text, pattern) == [199_000]


def test_border_and_repeating_unit():
    assert longest_prefix_also_suffix("level") == "l"
    assert longest_prefix_also_suffix("ababab") == "abab"
    assert longest_prefix_also_suffix("abc") == "" and longest_prefix_also_suffix("") == ""
    rnd = random.Random(3)
    for _ in range(400):
        s = rand_str(rnd, "ab", 12)
        expected = next(
            (s[:p] for p in range(1, len(s) + 1) if len(s) % p == 0 and s[:p] * (len(s) // p) == s),
            "",
        )
        assert shortest_repeating_unit(s) == expected


def test_longest_repeated_substring_against_brute_force():
    rnd = random.Random(4)
    for _ in range(300):
        s = rand_str(rnd, "abc", 14)
        best = 0
        for length in range(1, len(s)):
            subs = [s[i : i + length] for i in range(len(s) - length + 1)]
            if len(set(subs)) < len(subs):
                best = length
        got = longest_repeated_substring(s)
        assert len(got) == best
        if best:
            assert len(all_occurrences(s, got)) >= 2
    assert longest_repeated_substring("banana") == "ana"
