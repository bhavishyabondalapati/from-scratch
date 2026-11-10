# Curriculum

One folder per item. A box is ticked only when the folder has `impl.py`,
`test_impl.py`, `NOTES.md` (plus `PATTERNS.md` in Section 1) and its tests pass.

To resume work: find the first unticked box below, and read `PROGRESS.md` if it exists.

## Section 1 - Algorithms & data structures (`s1_algorithms/`)

- [x] `s1_algorithms/arrays_hashing` — arrays & hashing
- [x] `s1_algorithms/two_pointers` — two pointers
- [x] `s1_algorithms/sliding_window` — sliding window
- [x] `s1_algorithms/prefix_sums` — prefix sums
- [x] `s1_algorithms/binary_search` — binary search, incl. on the answer space
- [x] `s1_algorithms/stack_monotonic` — stack & monotonic stack
- [x] `s1_algorithms/queue_deque` — queue, deque, monotonic queue
- [x] `s1_algorithms/linked_list` — linked list operations
- [x] `s1_algorithms/lru_cache` — LRU cache
- [x] `s1_algorithms/lfu_cache` — LFU cache
- [x] `s1_algorithms/heap` — heap from scratch, top-k, merge k sorted, median of stream
- [x] `s1_algorithms/trees` — traversals (recursive + iterative), BST ops, LCA, serialize/deserialize
- [x] `s1_algorithms/trie` — trie
- [ ] `s1_algorithms/segment_tree` — segment tree
- [ ] `s1_algorithms/fenwick_tree` — Fenwick (binary indexed) tree
- [ ] `s1_algorithms/union_find` — union-find with rank + path compression
- [ ] `s1_algorithms/graph_traversal` — BFS, DFS, topological sort (Kahn + DFS), bipartite check
- [ ] `s1_algorithms/shortest_paths` — Dijkstra, Bellman-Ford, Floyd-Warshall
- [ ] `s1_algorithms/minimum_spanning_tree` — Prim, Kruskal
- [ ] `s1_algorithms/tarjan_scc` — Tarjan strongly connected components
- [ ] `s1_algorithms/backtracking` — subsets, permutations, combinations, N-Queens, sudoku
- [ ] `s1_algorithms/dp_1d_grid` — dynamic programming: 1D and 2D grid
- [ ] `s1_algorithms/dp_knapsack` — knapsack (0/1 + unbounded)
- [ ] `s1_algorithms/dp_sequences` — LIS (O(n log n)), LCS, edit distance
- [ ] `s1_algorithms/dp_interval_bitmask` — interval DP, bitmask DP
- [ ] `s1_algorithms/greedy_intervals` — greedy + intervals
- [ ] `s1_algorithms/sorting` — quick, merge, heap, counting, radix sort
- [ ] `s1_algorithms/quickselect` — quickselect
- [ ] `s1_algorithms/string_matching` — KMP, Rabin-Karp

## Section 2 - Math foundations (`s2_math/`)

- [ ] `s2_math/linalg` — vector/matrix operations
- [ ] `s2_math/gradient_descent` — batch GD, SGD, momentum, RMSProp, Adam
- [ ] `s2_math/gradient_check` — numerical gradient checking
- [ ] `s2_math/softmax_logsumexp` — softmax / log-sum-exp stability
- [ ] `s2_math/pca` — PCA via SVD and via eigendecomposition
- [ ] `s2_math/probability` — MLE and Bayes' rule

## Section 3 - Classical ML (`s3_classical_ml/`)

- [ ] `s3_classical_ml/linear_regression` — closed form + gradient descent
- [ ] `s3_classical_ml/ridge_lasso` — ridge, lasso (coordinate descent)
- [ ] `s3_classical_ml/logistic_regression` — binary + multinomial
- [ ] `s3_classical_ml/knn` — k-nearest neighbours
- [ ] `s3_classical_ml/naive_bayes` — Gaussian + multinomial
- [ ] `s3_classical_ml/decision_tree` — CART with gini + entropy
- [ ] `s3_classical_ml/random_forest` — random forest
- [ ] `s3_classical_ml/gradient_boosting` — regression + classification
- [ ] `s3_classical_ml/adaboost` — AdaBoost
- [ ] `s3_classical_ml/svm` — SVM (subgradient + SMO with kernels)
- [ ] `s3_classical_ml/kmeans` — k-means and k-means++
- [ ] `s3_classical_ml/gmm` — Gaussian mixture model with EM
- [ ] `s3_classical_ml/dbscan` — DBSCAN
- [ ] `s3_classical_ml/hierarchical_clustering` — agglomerative clustering
- [ ] `s3_classical_ml/pca` — PCA as a fit/transform model
- [ ] `s3_classical_ml/tsne` — t-SNE (small-scale)
- [ ] `s3_classical_ml/model_selection` — train/val/test split, k-fold CV, grid search
- [ ] `s3_classical_ml/metrics` — confusion matrix, precision/recall/F1, ROC-AUC, PR-AUC, calibration
- [ ] `s3_classical_ml/bias_variance` — bias-variance demo, regularization demo

## Section 4 - Deep learning (`s4_deep_learning/`)

NumPy only until the autograd engines are built; PyTorch allowed from `resnet_block` on.

- [ ] `s4_deep_learning/autograd_scalar` — scalar autograd engine (micrograd-style)
- [ ] `s4_deep_learning/autograd_tensor` — tensor autograd engine in NumPy
- [ ] `s4_deep_learning/mlp` — MLP with backprop, activations, Xavier/He init
- [ ] `s4_deep_learning/regularization` — dropout, weight decay, LR schedules
- [ ] `s4_deep_learning/normalization` — batch norm, layer norm
- [ ] `s4_deep_learning/cnn` — conv2d forward/backward (im2col), pooling, small CNN on digits
- [ ] `s4_deep_learning/rnn_lstm_gru` — RNN, LSTM, GRU forward/backward
- [ ] `s4_deep_learning/char_lm` — character-level language model
- [ ] `s4_deep_learning/resnet_block` — ResNet block (PyTorch)
- [ ] `s4_deep_learning/autoencoder` — autoencoder
- [ ] `s4_deep_learning/vae` — variational autoencoder
- [ ] `s4_deep_learning/gan` — small GAN on 2D toy data
- [ ] `s4_deep_learning/unet` — U-Net on a synthetic segmentation task
- [ ] `s4_deep_learning/siamese` — Siamese network with contrastive loss

## Section 5 - NLP (`s5_nlp/`)

- [ ] `s5_nlp/tokenization` — whitespace/regex, BPE from scratch, WordPiece overview
- [ ] `s5_nlp/bow_tfidf` — bag-of-words, TF-IDF
- [ ] `s5_nlp/ngram_lm` — n-gram language model with smoothing, perplexity
- [ ] `s5_nlp/word2vec` — skip-gram with negative sampling, cosine similarity + analogy test
- [ ] `s5_nlp/glove` — GloVe (small)
- [ ] `s5_nlp/attention` — scaled dot-product + multi-head attention
- [ ] `s5_nlp/positional_encoding` — sinusoidal + RoPE
- [ ] `s5_nlp/seq2seq_attention` — seq2seq with Bahdanau attention
- [ ] `s5_nlp/gpt` — decoder-only transformer on tiny-shakespeare, KV cache, sampling
- [ ] `s5_nlp/bert_classifier` — tiny BERT-style encoder for sentiment
- [ ] `s5_nlp/lora` — LoRA by hand on a linear layer, applied to the tiny GPT
- [ ] `s5_nlp/bm25` — BM25 from scratch
- [ ] `s5_nlp/dense_retrieval` — dense retrieval, vector index (brute force + IVF)
- [ ] `s5_nlp/rag` — minimal RAG loop with an offline extractive answerer
- [ ] `s5_nlp/eval_metrics` — BLEU, ROUGE-L, exact match, recall@k, MRR, NDCG

## Section 6 - Reinforcement learning (`s6_rl/`)

- [ ] `s6_rl/bandits` — epsilon-greedy, UCB, Thompson sampling
- [ ] `s6_rl/dynamic_programming` — value iteration, policy iteration
- [ ] `s6_rl/td_learning` — Q-learning, SARSA on a gridworld

## Finish

- [ ] README.md with table of contents
- [ ] Makefile (`make test`, `make test-section N`, `make lint`, `make format`)
- [ ] GitHub Actions workflow (ruff + pytest, slow tests skipped)
