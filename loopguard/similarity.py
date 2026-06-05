import math
from typing import List, Set, Dict

# Standard set of English stopwords to improve keyword search similarity accuracy
STOP_WORDS: Set[str] = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "arent",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cant",
    "cannot",
    "could",
    "couldnt",
    "did",
    "didnt",
    "do",
    "does",
    "doesnt",
    "doing",
    "dont",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "hadnt",
    "has",
    "hasnt",
    "have",
    "havent",
    "having",
    "he",
    "hed",
    "hell",
    "hes",
    "her",
    "here",
    "heres",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "hows",
    "i",
    "id",
    "ill",
    "im",
    "ive",
    "if",
    "in",
    "into",
    "is",
    "isnt",
    "it",
    "its",
    "itself",
    "lets",
    "me",
    "more",
    "most",
    "mustnt",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shant",
    "she",
    "shed",
    "shell",
    "shes",
    "should",
    "shouldnt",
    "so",
    "some",
    "such",
    "than",
    "that",
    "thats",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "theres",
    "these",
    "they",
    "theyd",
    "theyll",
    "theyre",
    "theyve",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "wasnt",
    "we",
    "wed",
    "well",
    "were",
    "weve",
    "werent",
    "what",
    "whats",
    "when",
    "whens",
    "where",
    "wheres",
    "which",
    "while",
    "who",
    "whos",
    "whom",
    "why",
    "whys",
    "with",
    "wont",
    "would",
    "wouldnt",
    "you",
    "youd",
    "youll",
    "youre",
    "youve",
    "your",
    "yours",
    "yourself",
    "yourselves",
}


def tokenize(text: str) -> List[str]:
    """Tokenize a string into alpha-numeric lowercase terms, filtering stopwords."""
    cleaned = "".join([c.lower() if c.isalnum() or c.isspace() else " " for c in text])
    return [w for w in cleaned.split() if w not in STOP_WORDS]


class TFIDFSimilarity:
    """A zero-dependency TF-IDF cosine similarity engine for streaming texts."""

    def __init__(self):
        self.doc_count = 0
        self.df: Dict[str, int] = {}  # Document frequency of vocabulary terms

    def fit_step(self, text: str):
        """Fit vocabulary with a new document (step text)."""
        tokens = set(tokenize(text))
        if not tokens:
            return
        self.doc_count += 1
        for token in tokens:
            self.df[token] = self.df.get(token, 0) + 1

    def _get_tf_idf(self, text: str) -> Dict[str, float]:
        """Convert a document into TF-IDF vector weights."""
        tokens = tokenize(text)
        if not tokens:
            return {}
        tf: Dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        tfidf: Dict[str, float] = {}
        for token, count in tf.items():
            df_val = self.df.get(token, 0)
            # Standard TF-IDF calculation with Laplace smoothing
            idf = math.log((1 + self.doc_count) / (1 + df_val)) + 1
            tfidf[token] = count * idf
        return tfidf

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate the Cosine similarity between TF-IDF representations of two strings."""
        vec1 = self._get_tf_idf(text1)
        vec2 = self._get_tf_idf(text2)
        if not vec1 or not vec2:
            return 0.0

        dot_product = sum(vec1[token] * vec2.get(token, 0.0) for token in vec1)
        norm1 = math.sqrt(sum(val**2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val**2 for val in vec2.values()))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot_product / (norm1 * norm2)
