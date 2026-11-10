# Bag-of-words and TF-IDF

## Intuition
**Bag of words:** describe a document by *which words appear and how often*, ignoring order — like
tipping the words of a page into a bag and counting them. Simple, fast, and surprisingly strong for
topic-level tasks (spam, news category, search).

**TF-IDF** fixes the obvious flaw that common words ("the", "and") dominate the counts. Weight each
word by how *rare* it is across the collection: a word that appears in every document says nothing
about any particular one; a word that appears in few documents is a strong signal.

## Core math (sklearn defaults, reproduced exactly)
- Tokens: lowercase, regex `\b\w\w+\b` (2+ word characters). Vocabulary sorted alphabetically.
- `tf(t, d)` = raw count (or `1 + ln tf` if sublinear).
- `idf(t) = ln((1 + n)/(1 + df(t))) + 1` with smoothing, `ln(n/df(t)) + 1` without.
- `tfidf = tf · idf`, then each document vector is L2-normalised, so cosine similarity is just a
  dot product.
- n-grams (`ngram_range=(1, 2)`) recover a little word order ("not good").

## Complexity
Fitting: O(total tokens). The matrix is `documents × vocabulary`, extremely sparse (real
implementations use sparse matrices; here dense arrays keep the code readable).

## Common pitfalls
- Fitting the vocabulary/idf on test data (leakage).
- Vocabulary explosion with n-grams; control with `min_df`, `max_df`, `max_features`.
- Losing negation and order ("not bad" vs "bad"); add bigrams or use embeddings.
- Words unseen at training time are silently ignored at test time.
- Dense matrices for big corpora run out of memory.

## When to use vs. alternatives
- Strong baselines for text classification (with logistic regression or naive Bayes), keyword
  search, and document similarity; interpretable features.
- BM25 (`s5_nlp/bm25`) is the better-tuned version for search ranking.
- Semantic similarity (synonyms, paraphrases) → dense embeddings (`word2vec`, `dense_retrieval`).

## Interview questions
1. **What does IDF do?** Down-weights terms that occur in many documents and up-weights rare,
   discriminative terms.
2. **Why L2-normalise TF-IDF vectors?** So long documents do not get larger scores just for being
   long, and cosine similarity becomes a dot product.
3. **Limitations of bag of words?** No word order, no semantics (synonyms are unrelated
   dimensions), huge sparse vectors.
4. **Why is sublinear tf useful?** The 20th occurrence of a word adds much less evidence than the
   first; log scaling reflects that.
5. **TF-IDF vs BM25?** BM25 saturates term frequency and normalises by document length with tunable
   parameters; it usually ranks better for search.
