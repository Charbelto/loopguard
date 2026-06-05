import math
from loopguard.similarity import tokenize, TFIDFSimilarity


def test_tokenize():
    # Normal text tokenization
    assert tokenize("hello world") == ["hello", "world"]

    # Case normalization
    assert tokenize("Hello World") == ["hello", "world"]

    # Punctuation and symbols removal
    assert tokenize("hello, world! #programming 2026") == [
        "hello",
        "world",
        "programming",
        "2026",
    ]

    # Stop-word removal
    assert tokenize("this is a test about a loop") == ["test", "loop"]

    # Empty string or whitespaces only
    assert tokenize("") == []
    assert tokenize("   \t\n  ") == []
    assert tokenize("about under above") == []  # only stop words


def test_tfidf_similarity_identical():
    sim_engine = TFIDFSimilarity()
    text = "call the web search tool to retrieve database content"

    # Fit the vocabulary
    sim_engine.fit_step(text)

    # Identical strings should have cosine similarity of 1.0
    val = sim_engine.cosine_similarity(text, text)
    assert math.isclose(val, 1.0, rel_tol=1e-5)


def test_tfidf_similarity_disjoint():
    sim_engine = TFIDFSimilarity()
    text1 = "call web search tool"
    text2 = "write output to file database"

    sim_engine.fit_step(text1)
    sim_engine.fit_step(text2)

    # Disjoint strings (no overlapping non-stopword tokens) should be 0.0
    val = sim_engine.cosine_similarity(text1, text2)
    assert val == 0.0


def test_tfidf_similarity_partial():
    sim_engine = TFIDFSimilarity()
    text1 = "search weather in chicago today"
    text2 = "search weather in london tomorrow"

    sim_engine.fit_step(text1)
    sim_engine.fit_step(text2)

    val = sim_engine.cosine_similarity(text1, text2)
    # They share 'search' and 'weather', so similarity should be between 0 and 1
    assert 0.0 < val < 1.0


def test_tfidf_similarity_empty():
    sim_engine = TFIDFSimilarity()
    text1 = "valid text query"
    text2 = ""

    sim_engine.fit_step(text1)
    sim_engine.fit_step(text2)

    # Empty string should not fail and should yield 0.0 similarity
    assert sim_engine.cosine_similarity(text1, text2) == 0.0
    assert sim_engine.cosine_similarity(text2, text2) == 0.0
