# k-means and k-means++

## Intuition
Drop k pins on the map. Every point joins its nearest pin. Move each pin to the centre of its
group. Repeat until nothing moves. The result is k compact, roughly round clusters.

The start matters: two pins dropped in the same blob may never separate. **k-means++** chooses
starting pins one at a time, preferring points far from the pins already placed, so they start
spread out.

## Core math
- Objective (inertia): `J = Σᵢ ‖xᵢ − μ_{c(i)}‖²`.
- Assignment step: `c(i) = argmin_k ‖xᵢ − μₖ‖²` (minimises J for fixed centres).
- Update step: `μₖ = mean of points in cluster k` (minimises J for fixed assignments).
- Each step can only decrease J, and there are finitely many partitions ⇒ convergence to a
  **local** minimum. Finding the global minimum is NP-hard.
- k-means++: next centre chosen with probability `∝ D(x)²` (squared distance to the nearest
  chosen centre). Guarantees expected cost within O(log k) of optimal.
- k-means is EM for a Gaussian mixture with equal, spherical covariances and hard assignments.

## Complexity
O(n · k · d) per iteration, times iterations, times `n_init`. Memory O(n·k) for the distance table.

## Common pitfalls
- Unscaled features dominate the distance.
- Choosing k: inertia always falls as k grows; use the elbow, silhouette score, or a downstream
  metric.
- Assumes convex, similarly sized, roughly spherical clusters; fails on rings and moons (use
  DBSCAN / spectral clustering) and on clusters of very different spread (use GMM).
- Sensitive to outliers (means get dragged); k-medoids is more robust.
- Empty clusters: handle them (keep the old centre or re-seed).
- Labels are arbitrary: compare clusterings with permutation-invariant scores (ARI, NMI).

## When to use vs. alternatives
- Fast, scalable default for compact clusters, vector quantisation, image colour compression,
  initialising GMMs, building IVF indexes (see `s5_nlp/dense_retrieval`).
- Soft assignments / elliptical clusters → GMM. Arbitrary shapes + noise → DBSCAN.
  A hierarchy of clusters → agglomerative clustering. Huge data → mini-batch k-means.

## Interview questions
1. **Does k-means always converge? To what?** Yes, in finitely many steps, to a local minimum of
   inertia; hence multiple restarts.
2. **Why k-means++?** Spread-out initial centres give faster convergence and provably better
   expected objective than uniform random starts.
3. **How do you pick k?** Elbow on inertia, silhouette, gap statistic, or business/downstream needs.
4. **What shapes can it not find?** Non-convex or very unequal-variance clusters.
5. **How does it relate to GMM/EM?** It is the hard-assignment, equal-spherical-covariance limit of
   EM for Gaussian mixtures.
