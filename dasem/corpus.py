"""corpus."""


from abc import ABCMeta

from six import with_metaclass, u


class Corpus(with_metaclass(ABCMeta)):
    """Abstract class for corpus."""

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
