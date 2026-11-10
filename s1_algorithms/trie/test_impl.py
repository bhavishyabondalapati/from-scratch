import random
import re

from .impl import Trie


def rand_word(rnd, max_len=5):
    return "".join(rnd.choice("abc") for _ in range(rnd.randint(0, max_len)))


def test_trie_matches_python_set_under_random_operations():
    rnd = random.Random(0)
    trie, ref = Trie(), set()
    for _ in range(5000):
        w, op = rand_word(rnd), rnd.choice(["insert", "insert", "delete", "query"])
        if op == "insert":
            assert trie.insert(w) == (w not in ref)
            ref.add(w)
        elif op == "delete":
            assert trie.delete(w) == (w in ref)
            ref.discard(w)
        else:
            assert trie.search(w) == (w in ref)
            assert trie.starts_with(w) == any(x.startswith(w) for x in ref)
            assert trie.count_prefix(w) == sum(x.startswith(w) for x in ref)
            assert trie.words_with_prefix(w) == sorted(x for x in ref if x.startswith(w))
    assert trie.count_prefix("") == len(ref)


def test_delete_prunes_unused_nodes():
    trie = Trie()
    trie.insert("car")
    trie.insert("cart")
    trie.delete("cart")
    assert trie.search("car") and not trie.starts_with("cart")
    trie.delete("car")
    assert trie.root.children == {}  # fully pruned


def test_prefix_is_not_a_word():
    trie = Trie()
    trie.insert("apple")
    assert not trie.search("app") and trie.starts_with("app")
    trie.insert("app")
    assert trie.search("app")


def test_wildcard_search_matches_regex():
    rnd = random.Random(1)
    words = {rand_word(rnd, 4) for _ in range(40)}
    trie = Trie()
    for w in words:
        trie.insert(w)
    for _ in range(500):
        pattern = "".join(rnd.choice("abc.") for _ in range(rnd.randint(0, 4)))
        expected = any(re.fullmatch(pattern, w) for w in words)
        assert trie.search_with_wildcard(pattern) == expected
