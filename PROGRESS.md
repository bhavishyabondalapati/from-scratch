# Progress

**Status:** complete. All 86 curriculum items and the finish tasks (README, Makefile, CI) are done and
committed. 637 tests (621 fast + 16 slow) pass.

## If you continue working on this repo
1. `conda activate from-scratch`
2. `make test-fast` (about 35 s) or `make test` (everything, including training runs).
3. New topic: create `<section>/<topic>/impl.py`, `test_impl.py`, `NOTES.md`, add a line to
   `CURRICULUM.md`, then `scripts/done.sh <section>/<topic> "<commit message>"`.

## Shared helpers
- `s2_math/gradient_check` — finite-difference gradient checks used throughout.
- `s2_math/softmax_logsumexp` — stable softmax/sigmoid/cross-entropy.
- `s3_classical_ml/knn` (`pairwise_sq_distances`), `s3_classical_ml/decision_tree` (weighted trees).
- `s4_deep_learning/torch_utils.py` — device selection (Apple `mps` ~20× faster for conv nets), digits.
- `data_utils.py` — cached tiny-shakespeare download into the gitignored `data/`.
- `s5_nlp/corpus.py` — the repo's NOTES.md files as a retrieval corpus.
- `s6_rl/gridworld.py` — cliff-walking environment.
