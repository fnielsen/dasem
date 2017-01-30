"""models.

Usage:
  dasem.models

"""


from __future__ import absolute_import, division, print_function

import logging

import gensim

from os.path import join, sep

from numpy import zeros

from .config import data_directory
from .utils import make_data_directory


WORD2VEC_FILENAME = 'word2vec.pkl.gz'


class Word2Vec(object):
    """Gensim Word2vec abstract class.

    Parameters
    ----------
    autosetup : bool, optional
        Determines whether the Word2Vec model should be autoloaded.
    logging_level : logging.ERROR or other, default logging.WARN
        Logging level.

    Description
    -----------
    Trained models can be saved and loaded via the `save` and `load` methods.

    """

    def __init__(self, autosetup=True):
        """Setup model."""
        self.logger = logging.getLogger(__name__ + '.Word2Vec')
        self.logger.addHandler(logging.NullHandler())

        self.model = None
        if autosetup:
            self.logger.info('Autosetup')
            try:
                self.load()
            except IOError:
                # The file is probably not there
                self.logger.info('Loading word2vec model from failed')
                self.train()
                self.save()

    def data_directory(self):
        """Return data directory.

        Raises
        ------
        err : NotImplementedError
            Always raised as a derived class should define it.

        """
        raise NotImplementedError('Define this in derived class')

    def full_filename(self, filename):
        """Return filename with full filename path."""
        if sep in filename:
            return filename
        else:
            return join(self.data_directory(), filename)

    def load(self, filename=WORD2VEC_FILENAME):
        """Load model from pickle file.

        This function is unsafe. Do not load unsafe files.

        Parameters
        ----------
        filename : str
            Filename of pickle file.

        """
        full_filename = self.full_filename(filename)
        self.logger.info('Trying to load word2vec model from {}'.format(
            full_filename))
        self.model = gensim.models.Word2Vec.load(full_filename)

    def make_data_directory(self):
        """Make data directory for LCC."""
        raise NotImplementedError('Define this in derived class')

    def save(self, filename=WORD2VEC_FILENAME):
        """Save model to pickle file.

        The Gensim load file is used which can also compress the file.

        Parameters
        ----------
        filename : str, optional
            Filename.

        """
        full_filename = self.full_filename(filename)
        self.make_data_directory()
        self.model.save(full_filename)

    def iterable_sentence_words(self):
        """Yield list of sentence words.

        Raises
        ------
        err : NotImplementedError
            Always raised as a derived class should define it.

        """
        raise NotImplementedError("Define this in derived class")

    def train(self, size=100, window=5, min_count=5, workers=4,
              translate_aa=True, translate_whitespaces=True, lower=True,
              stem=False):
        """Train Gensim Word2Vec model.

        Parameters
        ----------
        size : int, default 100
            Dimension of the word2vec space. Gensim Word2Vec parameter.
        window : int, default 5
            Word window size. Gensim Word2Vec parameter.
        min_count : int, default 5
            Minimum number of times a word must occure to be included in the
            model. Gensim Word2Vec parameter.
        workers : int, default 4
            Number of Gensim workers.
        translate_aa : bool, default True
            Translate double-a to 'bolle-aa'.
        translate_whitespaces : bool, default True
            Translate multiple whitespaces to single whitespaces
        lower : bool, default True
            Lower case the words.
        stem : bool, default False
            Apply word stemming. DanishStemmer from nltk is used.

        """
        self.logger.info(
            ('Training word2vec model with parameters: '
             'size={size}, window={window}, '
             'min_count={min_count}, workers={workers}').format(
                 size=size, window=window, min_count=min_count,
                 workers=workers))
        self.model = gensim.models.Word2Vec(
            self.iterable_sentence_words(),
            size=size, window=window, min_count=min_count,
            workers=workers)

    def doesnt_match(self, words):
        """Return odd word of list.

        This method forward the matching to the `doesnt_match` method in the
        Word2Vec class of Gensim.

        Parameters
        ----------
        words : list of str
            List of words represented as strings.

        Returns
        -------
        word : str
            Outlier word.

        Examples
        --------
        >>> w2v = Word2Vec()
        >>> w2v.doesnt_match(['svend', 'stol', 'ole', 'anders'])
        'stol'

        """
        return self.model.doesnt_match(words)

    def most_similar(self, positive=[], negative=[], topn=10,
                     restrict_vocab=None, indexer=None):
        """Return most similar words.

        This method will forward the similarity search to the `most_similar`
        method in the Word2Vec class in Gensim. The input parameters and
        returned result are the same.

        Parameters
        ----------
        positive : list of str
            List of strings with words to include for similarity search.
        negative : list of str
            List of strings with words to discount.
        topn : int
            Number of words to return

        Returns
        -------
        words : list of tuples
            List of 2-tuples with word and similarity.

        Examples
        --------
        >>> w2v = Word2Vec()
        >>> words = w2v.most_similar('studie')
        >>> len(words)
        10

        """
        return self.model.most_similar(
            positive, negative, topn, restrict_vocab, indexer)

    def similarity(self, word1, word2):
        """Return value for similarity between two words.

        Parameters
        ----------
        word1 : str
            First word to be compared
        word2 : str
            Second word.

        Returns
        -------
        value : float
            Similarity as a float value between 0 and 1.

        """
        return self.model.similarity(word1, word2)

    def word_vector(self, word):
        """Return feature vector for word.

        Parameters
        ----------
        word : str
            Word.

        Returns
        -------
        vector : np.array
            Array will values from Gensim's syn0. If the word is not in the
            vocabulary, then a zero vector is returned.

        """
        self.model.init_sims()
        try:
            vector = self.model[word]

            # Normalized:
            # vector = self.model.syn0norm[self.model.vocab[word].index, :]
        except KeyError:
            vector = zeros(self.model.vector_size)
        return vector


def main():
    """Handle command-line interface."""
    print('dasem.models')


if __name__ == '__main__':
    main()
