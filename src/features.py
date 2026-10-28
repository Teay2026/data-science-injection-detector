"""Text features with scikit-learn, used from iteration 3 (notebook 6) on."""

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

# Pieces of 3 and 4 characters. With 5 too, the features of 60 000 emails take about 2 GB.
CHAR_SIZES = (3, 4)
# 2^18 columns, as Spark's HashingTF by default.
N_COLUMNS = 2**18


def word_tfidf(max_features):
    """Words split on spaces, like the Spark tokenizer of notebooks 4 and 5, with the same pruning."""
    return TfidfVectorizer(token_pattern=r"\S+", lowercase=False, min_df=5, max_df=0.95, max_features=max_features)


def char_hashing():
    """Pieces of 3 and 4 characters, each put in one of N_COLUMNS columns by a hash function.

    No vocabulary to build: there are far too many different pieces to keep a list in memory.
    """
    return HashingVectorizer(
        analyzer="char", ngram_range=CHAR_SIZES, lowercase=False,
        n_features=N_COLUMNS, alternate_sign=False, norm=None, dtype=np.float32,
    )


def column_of(pieces):
    """Column of each piece in char_hashing(), to read which pieces a model uses."""
    single = HashingVectorizer(analyzer=lambda piece: [piece], n_features=N_COLUMNS, alternate_sign=False, norm=None)
    return single.transform(pieces).indices
