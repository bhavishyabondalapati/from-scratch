# Trie (prefix tree)

## Intuition
Store words letter by letter along paths of a tree. Words with a common beginning share the
beginning of their path, like a dictionary's guide words. To check whether any word starts with
"pre", just follow p → r → e; you never look at the other words at all.

## Core idea
- Each node has a map `char -> child` and a flag "a word ends here".
- `insert` / `search` / `starts_with` all walk one node per character.
- `pass_count` on each node (how many words go through it) gives prefix counts in O(L) and makes
  deletion easy: decrement along the path and cut the branch when a count hits 0.

## Complexity
`L` = word length, `N` = number of words, `Σ` = alphabet size.

| Operation | Time |
|---|---|
| insert / search / starts_with / count_prefix / delete | O(L) |
| words_with_prefix | O(L + size of the matching subtree) |
| wildcard search | O(L) with no dots, up to O(Σ^dots · L) |
| Space | O(total characters) worst case; node overhead is significant |

## Common pitfalls
- Confusing "is a prefix" with "is a word" (need the `is_word` flag).
- Memory: one dict per node is heavy; a fixed array of 26 is faster but wasteful for sparse nodes.
- Deleting without pruning (leaks nodes) or pruning nodes still used by other words.
- Inserting a duplicate and double-counting `pass_count`.

## When to use vs. alternatives
- Trie: prefix queries, autocomplete, word-search on a grid, longest common prefix, IP routing
  (bitwise trie), max XOR pair.
- Hash set: exact membership only — simpler, less memory.
- Sorted array + binary search: prefix ranges on static data with far less memory.
- Radix/compressed trie: merges single-child chains to cut memory.

## Interview questions
1. **Trie vs hash set for a dictionary?** Both O(L) lookup; the trie also answers prefix
   questions and shares prefixes, the hash set has less overhead and better cache behaviour.
2. **How would you implement autocomplete with ranking?** Walk to the prefix node, then DFS; or
   cache the top-k completions at each node for O(L) queries.
3. **How do you delete a word safely?** Track pass-through counts (or check for other children /
   `is_word` while unwinding) and remove only nodes no other word uses.
4. **How do tries help the grid word-search problem?** Search all words at once: DFS the grid
   while walking the trie, pruning as soon as the current path is not a prefix of any word.
5. **Space optimisation?** Compressed (radix) tries, arrays instead of dicts, or a DAWG that also
   shares suffixes.
