# Progress

**Status:** Sections 1–3 complete and committed; all tests pass.
**Resume at:** `s4_deep_learning/autograd_scalar` (first unticked box in `CURRICULUM.md`).

## How to resume
1. `conda activate from-scratch`
2. `pytest -q` — everything should pass before you start.
3. Open `CURRICULUM.md`, take the first `[ ]` item, build `impl.py`, `test_impl.py`, `NOTES.md`.
4. Finish an item with `scripts/done.sh <folder> "<commit message>"` (formats, lints, tests,
   ticks the box, commits).
5. At the end of each section run `pytest -q <section folder>`.

## Conventions already in place
- Shared helpers to reuse: `s2_math/gradient_check` (finite-difference gradient checks),
  `s2_math/softmax_logsumexp` (stable softmax/sigmoid/cross-entropy), `s3_classical_ml/knn`
  (`pairwise_sq_distances`), `s3_classical_ml/decision_tree` (trees with sample weights).
- Section 4: NumPy only until both autograd engines exist; PyTorch allowed from
  `resnet_block` on. Every DL item saves a training-curve PNG and states the final metric in NOTES.md.
- Mark training tests that take more than a few seconds with `@pytest.mark.slow`.
