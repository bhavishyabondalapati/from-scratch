# DBSCAN

## Intuition
Clusters are **crowded regions** separated by empty space. Pick a radius `eps`. A point with at
least `min_samples` neighbours inside that radius is in a crowd (a *core* point). Crowds that
touch merge into one cluster, so a cluster can snake around in any shape. Points near a crowd but
not crowded themselves join as *border* points. Everything else is *noise*.

You never say how many clusters there are; DBSCAN discovers it.

## Core math
- ε-neighbourhood: `N(p) = {q : ‖p − q‖ ≤ ε}` (includes p).
- Core point: `|N(p)| ≥ min_samples`.
- Directly density-reachable: q ∈ N(p) and p is core. A cluster = all points density-reachable
  from a core point (BFS/DFS over core points).
- Border points may be reachable from two clusters; which one claims them depends on visit order
  (core points and noise are deterministic).

## Complexity
Brute force: O(n²) time and memory for the neighbour table. With a spatial index (KD-tree,
ball tree) roughly O(n log n) in low dimensions.

## Common pitfalls
- Choosing `eps`: plot each point's distance to its k-th nearest neighbour (k = min_samples),
  sorted, and pick the "knee".
- One global `eps` cannot handle clusters of very different densities (use HDBSCAN/OPTICS).
- Feature scaling changes every distance.
- High dimensions: distances concentrate and density loses meaning.
- Border point assignment is order-dependent.

## When to use vs. alternatives
- Arbitrary-shaped clusters, unknown number of clusters, outlier detection, geospatial data.
- Compact spherical clusters → k-means (faster). Varying density → HDBSCAN. Soft/probabilistic
  → GMM.

## Interview questions
1. **Core, border, noise?** Core: ≥ min_samples within eps. Border: within eps of a core point
   but not core. Noise: neither.
2. **Advantages over k-means?** No k needed, any shape, handles outliers explicitly.
3. **How to choose eps and min_samples?** min_samples ≈ 2·d or domain-based; eps from the
   k-distance plot knee.
4. **Main weakness?** A single density threshold fails for clusters of different densities.
5. **Is DBSCAN deterministic?** Core and noise points yes; border points can depend on order.
