"""text."""


from nltk import sent_tokenize
from nltk import word_tokenize as nltk_word_tokenize


def sentence_tokenize(text):
    """Tokenize a Danish text into sentence.

    The model from NTLK trained on Danish is used.

    Parameters
    ----------
    text : str
        The text to be tokenized.

    Returns
    -------
    sentences : list of str
        Sentences as list of strings.

    Examples
    --------
    >>> text = 'Hvad!? Hvor har du f.eks. siddet?'
    >>> sentences = sentence_tokenize(text)
    >>> sentences
    ['Hvad!?', 'Hvor har du f.eks. siddet?']

    """
    return sent_tokenize(text, language='danish')


def word_tokenize(sentence):
    """Tokenize a Danish sentence into words."""
    return nltk_word_tokenize(sentence)
