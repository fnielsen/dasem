
import pytest

from dasem.fullmonty import Word2Vec


@pytest.fixture
def w2v():
    return Word2Vec()


def test_w2v(w2v):
    word_and_similarities = w2v.most_similar('dreng')
    assert len(word_and_similarities) == 10
