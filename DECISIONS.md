# Decisions

Choices made while building the repo, and why.

## Layout

- **Section folders are `s1_algorithms`, `s2_math`, ...** The `sN_` prefix keeps them in
  curriculum order, is a valid Python package name (a bare `1_algorithms` is not), and lets
  `make test-section N` find the folder with a glob. `math/` was avoided because it would
  shadow the standard library.
- **Each bullet in the brief became one or more topic folders.** Big bullets were split so
  each `impl.py` stays focused (for example graphs became `graph_traversal`,
  `shortest_paths`, `minimum_spanning_tree`, `tarjan_scc`). The mapping is in `CURRICULUM.md`.
- **Every folder is a package (`__init__.py`)** so each test can do `from .impl import ...`
  even though all the files share the names `impl.py` / `test_impl.py`.
- **`scripts/done.sh`** formats, lints, tests, ticks the curriculum box and commits one
  topic. It is the "definition of done" for an item.

## Dependencies

- Implementations: standard library + NumPy. PyTorch appears only from
  `s4_deep_learning/resnet_block` onwards (after both autograd engines exist).
- scikit-learn / SciPy / PyTorch are imported **only inside test files** before that point.
- matplotlib is used only to save training-curve PNGs (it is not part of any algorithm).
- No `hypothesis`: property tests are plain loops over seeded `random` inputs, which keeps
  the dependency list short and the tests easy to read.

## Testing

- Reference comparisons use a tolerance appropriate to the method: exact solvers
  (closed-form regression, PCA) match to ~1e-6; iterative or randomised methods
  (forests, boosting, SVM) are checked on accuracy against the reference and a trivial baseline.
- Every hand-written backward pass is checked against central finite differences.
- Tests that train for more than a few seconds are marked `@pytest.mark.slow` and
  skipped in CI (`pytest -m "not slow"`).

## Lint

- Ruff rules `E, F, I, UP, B`. `E741` is ignored because math code naturally uses names
  like `l` and `I`; `E501` is ignored because the formatter handles code width and long
  comment lines are fine.

## Ordering

- `s5_nlp/eval_metrics` was built before `bm25`, `dense_retrieval` and `rag` (it is listed last in
  the curriculum) because those three topics use its recall@k / MRR / exact-match functions.

## Datasets chosen to keep everything offline

- Analogies (`word2vec`, `glove`): a synthetic corpus with planted attributes, so the right answers
  are known exactly.
- Sentiment (`bert_classifier`): generated reviews whose labels depend on negation and "but"
  clauses, so word order matters and bag-of-words is a meaningful baseline.
- Retrieval and RAG: the repository's own NOTES.md files, split into sections, frozen in
  `s5_nlp/corpus_snapshot.json`. Without the snapshot, editing any note (including the notes that
  report retrieval numbers) changed the corpus and the reported metrics. Regenerate with
  `python -m s5_nlp.corpus`.
- tiny-shakespeare is the only download (cached in the gitignored `data/` folder).
