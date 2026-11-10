# Minimal RAG loop with an offline extractive answerer

**Setup:** the corpus is this repository's NOTES.md files (section passages). The ~410 interview
questions are **held out** of the index and their written answers are the references. Pipeline: BM25
top-5 passages → score every sentence by idf-weighted overlap with the question plus a bonus for its
passage's rank → return the best sentence **with a citation**. No language model, no API.

**Final metric** (see [`results.txt`](results.txt) for numbers and example outputs):

| | ROUGE-L F vs reference answer |
|---|---|
| random corpus sentence | 0.012 |
| first sentence of the top retrieved passage | 0.056 |
| **RAG sentence selection** | **0.098** (token F1 0.10) |

The cited passage comes from the question's own topic for **57%** of questions, which is essentially the
retriever's ceiling: with BM25 the top-1 passage is from the right topic about 58% of the time, and no
re-weighting of the answerer moved it.

**Honest reading:** the loop is correct and clearly better than baselines, but answers are weak. The
reference answers are paraphrases written for the interview section, and an extractive system can only
quote existing sentences. It also picks keyword-heavy fragments (e.g. a complexity-table row for "why is
lookup O(1)?"). The usual fixes are a reranker and an abstractive generator (an LLM, or the tiny GPT if it
were trained on QA data) that reads the retrieved passages.

## Intuition
A language model only knows what it saw in training. **Retrieval-augmented generation** looks things up
first: find the passages most relevant to the question, then produce an answer *grounded* in them, with
sources. Updating knowledge means updating the index, not retraining the model.

Here the "generator" is extractive (it selects a sentence), which keeps everything offline and makes every
answer verifiable: it is a quote, and the citation says where it came from.

## Core math / algorithm
1. Retrieve: BM25 top-k (`s5_nlp/bm25`).
2. Candidate sentences from those passages.
3. Score: `Σ_{t ∈ q ∩ s} idf(t) / Σ_{t ∈ q} idf(t) + w · score(passage)/score(top passage)`, w = 1.
4. Answer = best sentence + passage id. Unanswerable (nothing retrieved) → explicit "not found".
5. Evaluate retrieval (topic hit) and answer quality (ROUGE-L, token F1 from `s5_nlp/eval_metrics`).

## Complexity
Retrieval O(posting lists); answer selection O(k · sentences · |q|). Index built once, O(corpus).

## Common pitfalls
- Leakage: indexing the very text the references come from (the interview sections are excluded here).
- Chunking: passages too long dilute relevance; too short lose context.
- Garbage in: the answer can only be as good as retrieval (the ceiling above).
- Ungrounded generation: an LLM generator may ignore the retrieved text; always show citations and check them.
- Evaluating only with n-gram overlap: paraphrased correct answers score low.

## When to use vs. alternatives
- Question answering over private or changing documents, support bots, internal search with answers.
- Production: hybrid retrieval + cross-encoder reranking + an LLM generator instructed to cite and to
  say "not found". Fine-tuning instead of RAG when the knowledge is stable and style matters more than facts.

## Interview questions
1. **Why RAG instead of fine-tuning for knowledge?** Up-to-date and private knowledge without retraining,
   citations, and fewer hallucinations about facts in the corpus.
2. **What are the main failure modes?** Retrieval misses, poor chunking, the generator ignoring or
   misreading context, and context windows overflowing.
3. **How do you evaluate a RAG system?** Separately: retrieval (recall@k, MRR) and answer quality
   (exact match/F1, ROUGE, faithfulness to sources, human or model-based judging).
4. **Extractive vs abstractive answering?** Extractive quotes spans (verifiable, limited); abstractive
   writes new text (fluent, can hallucinate).
5. **How would you improve this system?** Better retrieval (hybrid + reranker), smarter chunking, and an
   abstractive generator conditioned on the retrieved passages.
