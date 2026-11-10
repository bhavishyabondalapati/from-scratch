# Evaluation metrics: BLEU, ROUGE-L, exact match, recall@k, MRR, NDCG

These metrics are used by `bm25`, `dense_retrieval` and `rag`.

## Intuition
- **BLEU** (translation): what fraction of the output's 1-, 2-, 3- and 4-word chunks appear in a
  reference translation? Plus a penalty for outputs that are too short. Precision-oriented.
- **ROUGE-L** (summarisation): how long is the longest sequence of words that appears, in order, in
  both the output and the reference? Recall-oriented.
- **Exact match / token F1** (question answering): is the answer string right after normalising case,
  punctuation and articles? F1 gives partial credit for overlapping words.
- **Recall@k:** did the relevant documents make it into the top k?
- **MRR:** how high is the *first* relevant result? (1 for rank 1, ½ for rank 2, ...)
- **NDCG:** graded relevance with a logarithmic discount for lower positions, divided by the best
  possible score.

## Core math
- BLEU = `BP · exp((1/N) Σₙ log pₙ)`, `pₙ` = clipped n-gram precision (each n-gram counted at most as
  often as in the best reference), `BP = min(1, exp(1 − r/c))` with r the closest reference length.
- ROUGE-L: `P = LCS/|hyp|`, `R = LCS/|ref|`, `F = (1 + β²)PR / (R + β²P)`.
- EM / F1: SQuAD normalisation, F1 over token multisets.
- `recall@k = |top-k ∩ relevant| / |relevant|`, `RR = 1/rank_first_relevant`, MRR = mean RR.
- `DCG@k = Σᵢ relᵢ / log₂(i + 1)`, `NDCG = DCG / IDCG` (matches `sklearn.metrics.ndcg_score`, tested).

## Complexity
BLEU O(N · L) with hashing; ROUGE-L O(L_hyp · L_ref) (LCS DP); ranking metrics O(k).

## Common pitfalls
- Sentence-level BLEU of 0 because one n-gram order has no match (use smoothing, or corpus BLEU).
- Different tokenisation/normalisation makes BLEU numbers incomparable (sacreBLEU standardises it).
- n-gram metrics reward surface overlap: correct paraphrases score low, fluent nonsense can score well.
- Recall@k with k ≥ corpus size is trivially 1.
- NDCG with all-zero relevance (undefined; treat as 0).

## When to use vs. alternatives
- BLEU: machine translation benchmarks. ROUGE: summarisation. EM/F1: extractive QA.
- Ranking: recall@k for first-stage retrieval (what reaches the reranker), MRR when one answer
  matters, NDCG for graded relevance.
- Open-ended generation: human evaluation, model-based judges, embedding similarity (BERTScore).

## Interview questions
1. **Why does BLEU clip n-gram counts?** So repeating a common word cannot inflate precision.
2. **Why the brevity penalty?** Precision alone rewards very short outputs.
3. **BLEU vs ROUGE?** BLEU is precision-oriented over n-grams; ROUGE(-L) is recall-oriented and LCS
   allows gaps.
4. **MRR vs NDCG?** MRR only cares about the first relevant hit; NDCG uses all hits with graded
   relevance and position discounting.
5. **Limitations of n-gram metrics?** Insensitive to meaning; penalise valid paraphrases; correlate
   weakly with human judgement for open-ended text.
