"""corpus."""


from abc import ABCMeta

import logging

from nltk.stem.snowball import DanishStemmer
from nltk.tokenize import WordPunctTokenizer

from six import with_metaclass, u


class Corpus(with_metaclass(ABCMeta)):
    """Abstract class for corpus."""

    def __init__(self):
        """Set up tokenizer."""
        self.logger = logging.getLogger(__name__ + '.Corpus')
        self.logger.addHandler(logging.NullHandler())

        self.logger.debug('Setup word tokenizer')
        self.word_tokenizer = WordPunctTokenizer()

        self.logger.debug('Setup stemmer')
        self.stemmer = DanishStemmer()

    def iter_sentence_words(self, lower=True, stem=False):
        """Yield list of words from sentences.

        Parameters
        ----------
        lower : bool, default True
            Lower case the words.
        stem : bool, default False
            Apply word stemming. DanishStemmer from nltk is used.

        Yields
        ------
        words : list of str
            List of words

        """
        for sentence in self.iter_sentences():
            words = self.word_tokenizer.tokenize(sentence)
            if lower:
                words = [word.lower() for word in words]
            if stem:
                words = [self.stemmer.stem(word) for word in words]

            yield words

    def iter_tokenized_sentences(self, lower=True, stem=False):
        """Yield string with tokenized sentences.

        Parameters
        ----------
        lower : bool, default True
            Lower case the words.
        stem : bool, default False
            Apply word stemming. DanishStemmer from nltk is used.

        Yields
        ------
        tokenized_sentence : str
            Sentence as string with tokens separated by a whitespace.

        """
        for words in self.iter_sentence_words(lower=lower, stem=stem):
            tokenized_sentence = u(" ").join(words)
            yield tokenized_sentence
