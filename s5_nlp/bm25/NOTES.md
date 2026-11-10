# BM25

**Final metric:** on this repository's own notes (401 section passages from 80 topics; the 400
interview questions are held out as queries, and a passage counts as relevant if it comes from the
question's topic): **BM25 MRR 0.69**, with a relevant passage in the top 10 for **90%** of questions.
TF-IDF cosine similarity gets MRR 0.63; random ranking would be about 0.01–0.02. Tuning k1/b barely
matters here (0.69–0.70). Corpus loader: [`s5_nlp/corpus.py`](../corpus.py).

## Intuition
BM25 is the scoring function behind most keyword search engines (Lucene, Elasticsearch). It ranks a
document higher when it contains the query's words, especially **rare** words, with two corrections
to plain term counting:
- **Saturation:** the first time a document says "kneser" is strong evidence; the 20th adds little.
- **Length normalisation:** a long document will contain more matches by chance, so matches in a
  short document count for more.

## Core math
`score(q, d) = Σ_{t ∈ q} idf(t) · tf·(k₁ + 1) / (tf + k₁·(1 − b + b·|d|/avgdl))`
- `idf(t) = ln(1 + (N − df + 0.5)/(df + 0.5))` (always positive).
- `k₁` ∈ [1.2, 2]: how quickly tf saturates (the tf factor is bounded by `k₁ + 1`, tested).
  `k₁ = 0` ignores tf entirely.
- `b` ∈ [0, 1]: strength of length normalisation (`b = 0` disables it, tested).
- Inverted index: `term → {doc: tf}`, so a query touches only documents containing its terms.
  Scores match a direct implementation of the formula (tested).

## Complexity
Indexing O(total tokens). Query O(Σ posting-list lengths of the query terms), far smaller than the
corpus for selective terms. Production engines add skip lists and top-k pruning (WAND, MaxScore).

## Common pitfalls
- Tokenisation mismatch between index and query (case, stemming, stopwords).
- Vocabulary mismatch: "car" never matches "automobile" (dense retrieval fixes this; see
  `dense_retrieval`; or combine both in a hybrid).
- Classic idf goes negative for terms in more than half the documents; use the +1 form.
- Very short documents dominate if b is too high.

## When to use vs. alternatives
- Strong, cheap, explainable first-stage retrieval; works with no training data; great for exact
  terms (names, codes, error messages).
- Semantic matching / paraphrases → dense embeddings. Best results usually come from BM25 + dense
  hybrid retrieval followed by a cross-encoder reranker.

## Interview questions
1. **How does BM25 differ from TF-IDF?** Saturating term frequency and explicit document-length
   normalisation, with tunable k₁ and b.
2. **What do k₁ and b control?** k₁ how fast repeated terms stop adding score; b how strongly long
   documents are penalised.
3. **Why an inverted index?** Scoring only documents containing a query term makes search
   sub-linear in corpus size.
4. **When does BM25 beat dense retrieval?** Rare exact terms, out-of-domain data, identifiers; no
   training needed.
5. **How would you evaluate a retriever?** Recall@k (does the answer reach the reranker), MRR/NDCG for
   ranking quality, on held-out queries with relevance labels.
