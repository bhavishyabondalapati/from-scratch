# Hierarchical (agglomerative) clustering

## Intuition
Start with every point as its own cluster. Repeatedly merge the two closest clusters until only
one is left. The history of merges is a tree (a **dendrogram**): cut it at any height to get any
number of clusters, without re-running anything.

"Closest clusters" needs a definition (the **linkage**):
- **single:** distance between their closest points — follows chains, finds snakes, but can
  link distinct groups through a thin bridge of points.
- **complete:** distance between their farthest points — compact, similar-diameter clusters.
- **average:** mean distance over all pairs — a compromise.
- **ward:** merge the pair that increases total within-cluster variance the least — k-means-like
  round clusters.

## Core math
Lance-Williams update: after merging i and j, the distance to any other cluster k is a formula
of the old distances only:
- single `min(d_ik, d_jk)`; complete `max(d_ik, d_jk)`;
  average `(nᵢ d_ik + nⱼ d_jk)/(nᵢ + nⱼ)`;
- ward `√[((nᵢ+n_k)d_ik² + (nⱼ+n_k)d_jk² − n_k d_ij²) / (nᵢ+nⱼ+n_k)]`.
Single linkage = the minimum spanning tree with its longest edges removed (Kruskal order).

## Complexity
Naive: O(n³) time, O(n²) memory (the version here). Priority-queue versions O(n² log n);
single linkage via MST O(n²). Not practical beyond ~10⁴ points.

## Common pitfalls
- Quadratic memory for the distance matrix.
- Greedy merges are final; an early bad merge is never undone.
- Single linkage "chaining"; Ward only makes sense with Euclidean distance.
- Ties in distances make the tree non-unique.
- Scale your features.

## When to use vs. alternatives
- Small/medium data, need a hierarchy (taxonomies, phylogenetics, document topics), or want to
  explore several k at once.
- Large data → k-means / mini-batch k-means. Density-shaped clusters with noise → DBSCAN/HDBSCAN.

## Interview questions
1. **Agglomerative vs divisive?** Bottom-up merging vs top-down splitting; agglomerative is
   far more common.
2. **How do linkages differ?** Single = nearest points (chaining), complete = farthest (compact),
   average = mean, ward = variance increase.
3. **How do you choose the number of clusters?** Cut the dendrogram at a large gap in merge
   heights, or use a validity index.
4. **Complexity?** O(n²) memory and O(n²)–O(n³) time.
5. **Relation to MSTs?** Single-linkage clustering is equivalent to building an MST and
   removing its k − 1 heaviest edges.
