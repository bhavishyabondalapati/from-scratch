import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer as SkCount
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf
from sklearn.linear_model import LogisticRegression

from .impl import CountVectorizer, TfidfVectorizer, cosine_similarity_matrix, idf_by_hand, top_terms

DOCS = [
    "The cat sat on the mat.",
    "The dog sat on the log, and the dog barked.",
    "Cats and dogs: natural enemies? Not always!",
    "A quick brown fox jumps over the lazy dog dog dog.",
    "Stock markets rallied as tech stocks surged.",
    "The central bank raised interest rates again.",
]


@pytest.mark.parametrize(
    "ngram_range, min_df, binary", [((1, 1), 1, False), ((1, 2), 1, False), ((1, 1), 2, True)]
)
def test_counts_match_sklearn(ngram_range, min_df, binary):
    ours = CountVectorizer(ngram_range, min_df, binary)
    ref = SkCount(ngram_range=ngram_range, min_df=min_df, binary=binary)
    np.testing.assert_array_equal(ours.fit_transform(DOCS), ref.fit_transform(DOCS).toarray())
    assert ours.vocabulary_ == ref.vocabulary_
    new = ["the cat and the dog and an unseen word"]
    np.testing.assert_array_equal(ours.transform(new), ref.transform(new).toarray())


@pytest.mark.parametrize("sublinear, smooth", [(False, True), (True, True), (False, False)])
def test_tfidf_matches_sklearn(sublinear, smooth):
    ours = TfidfVectorizer(sublinear_tf=sublinear, smooth_idf=smooth)
    ref = SkTfidf(sublinear_tf=sublinear, smooth_idf=smooth)
    np.testing.assert_allclose(ours.fit_transform(DOCS), ref.fit_transform(DOCS).toarray())
    np.testing.assert_allclose(ours.idf_, ref.idf_)


def test_idf_weights_rare_terms_higher():
    vec = TfidfVectorizer().fit(DOCS)
    idf = dict(zip(sorted(vec.vocabulary_), vec.idf_))
    assert idf["the"] < idf["dog"] < idf["fox"]  # "the" is everywhere, "fox" is in one doc
    assert idf["fox"] == pytest.approx(idf_by_hand("fox", DOCS))
    rows = vec.transform(DOCS)
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1)
    assert top_terms(vec, rows[3], 1) == ["dog"]  # repeated three times in that document


def test_bag_of_words_ignores_order():
    vec = CountVectorizer().fit(["dog bites man"])
    np.testing.assert_array_equal(
        vec.transform(["dog bites man"]), vec.transform(["man bites dog"])
    )
    bigrams = CountVectorizer((1, 2)).fit(["dog bites man", "man bites dog"])
    assert not np.array_equal(
        bigrams.transform(["dog bites man"]), bigrams.transform(["man bites dog"])
    )


def test_retrieval_and_classification():
    vec = TfidfVectorizer().fit(DOCS)
    X = vec.transform(DOCS)
    q = vec.transform(["interest rates and the bank"])
    assert int(np.argmax(cosine_similarity_matrix(q, X))) == 5
    # Two topics (animals vs finance), generated from separate word lists.
    rng = np.random.default_rng(0)
    animals = "cat dog fox bark meow fur paw tail pet vet leash kennel".split()
    finance = "stock bond bank rate yield market fund price trade loan debt".split()
    common = "the a of and is on in today very big".split()

    def doc(topic):
        return " ".join(rng.choice(topic, 4).tolist() + rng.choice(common, 8).tolist())

    texts = [doc(animals) for _ in range(100)] + [doc(finance) for _ in range(100)]
    labels = np.array([0] * 100 + [1] * 100)
    idx = rng.permutation(200)
    tr, te = idx[:140], idx[140:]
    vec = TfidfVectorizer().fit([texts[i] for i in tr])
    clf = LogisticRegression().fit(vec.transform([texts[i] for i in tr]), labels[tr])
    assert clf.score(vec.transform([texts[i] for i in te]), labels[te]) > 0.95  # baseline 0.5
