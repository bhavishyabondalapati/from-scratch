import random
from pathlib import Path

import numpy as np

from s5_nlp.bm25.impl import tokenize
from s5_nlp.corpus import Passage, load_corpus
from s5_nlp.eval_metrics.impl import rouge_l, token_f1

from .impl import RAG, split_sentences

TOY = [
    Passage(
        "geo#1",
        "geo",
        "Geography",
        "Paris is the capital of France. It lies on the river Seine in the north.",
    ),
    Passage(
        "geo#2",
        "geo",
        "Geography",
        "Mount Everest is the highest mountain on Earth at about 8849 metres.",
    ),
    Passage(
        "bio#1",
        "bio",
        "Biology",
        "Mitochondria produce most of the chemical energy needed by the cell.",
    ),
]


def test_answers_come_from_retrieved_passages_with_citations():
    rag = RAG(TOY, k=2)
    a = rag.answer("What is the capital of France?")
    assert a.source == "geo#1" and "Paris is the capital of France" in a.text
    assert a.source in a.retrieved
    assert "8849" in rag.answer("How high is mount everest?").text
    assert rag.answer("which organelle produces energy for the cell").source == "bio#1"
    assert "[source: geo#1]" in rag.format("capital of France")


def test_unanswerable_question_is_handled():
    a = RAG(TOY).answer("zzzz qqqq")
    assert a.source == "" and a.retrieved == []


def test_sentence_splitting():
    s = split_sentences(
        "First sentence is here today. Second one is also here. ok | a table cell with words"
    )
    assert s == [
        "First sentence is here today.",
        "Second one is also here.",
        "a table cell with words",
    ]


def test_rag_on_the_repository_notes():
    passages, queries = load_corpus()
    rag = RAG(passages, k=5)
    rnd = random.Random(0)
    all_sentences = [s for p in passages for s in split_sentences(p.text)]
    rows, ours, rand, top1 = [], [], [], []
    for q in queries:
        a = rag.answer(q.text)
        ref = tokenize(q.answer)
        ours.append(rouge_l(tokenize(a.text), ref)["f"])
        rand.append(rouge_l(tokenize(rnd.choice(all_sentences)), ref)["f"])
        hits = rag.retrieve(q.text)
        first = (split_sentences(hits[0][0].text) if hits else []) or [""]
        top1.append(rouge_l(tokenize(first[0]), ref)["f"])
        rows.append((q, a, token_f1(a.text, q.answer)))
    topic_hit = np.mean([a.source.split("#")[0] == q.topic for q, a, _ in rows])
    assert topic_hit > 0.5  # cited passage from the right topic (retrieval caps this near 0.58)
    assert np.mean(ours) > 2 * np.mean(rand)
    assert np.mean(ours) > np.mean(top1)  # choosing the sentence beats "first sentence of top hit"
    examples = "\n\n".join(
        rag.format(q.text) + f"\n(reference: {q.answer})" for q, _, _ in rows[::80]
    )
    (Path(__file__).parent / "results.txt").write_text(
        f"{len(queries)} held-out interview questions\n"
        f"cited passage from the correct topic: {topic_hit:.3f}\n"
        f"ROUGE-L F vs reference answer: RAG {np.mean(ours):.3f}, first sentence of top passage "
        f"{np.mean(top1):.3f}, random sentence {np.mean(rand):.3f}\n"
        f"token F1: {np.mean([f for *_, f in rows]):.3f}\n\nExamples:\n\n{examples}\n"
    )
