"""models.

Usage:
  dasem.models

"""


from __future__ import absolute_import, division, print_function

from abc import ABCMeta

import logging

import gensim

from os.path import join, sep

from numpy import argsort, array, dot, newaxis, sqrt, zeros

from six import with_metaclass

import fasttext

from .utils import make_data_directory


WORD2VEC_FILENAME = 'word2vec.pkl.gz'

DOC2VEC_FILENAME = 'doc2vec.pkl.gz'

SENTENCES_FILENAME = 'sentences.txt'

FAST_TEXT_SKIPGRAM_MODEL_FILENAME = 'fasttext-skipgram-model'
FAST_TEXT_CBOW_MODEL_FILENAME = 'fasttext-cbow-model'


class Doc2Vec(with_metaclass(ABCMeta)):
    """Gensim Doc2vec for a corpus."""

    def __init__(self, autosetup=True):
        """Set up model.

        Parameters
        ----------
        autosetup : bool, optional
            Determines whether the DocVec model should be autoloaded.

        """
        self.logger = logging.getLogger(__name__ + '.Doc2Vec')
        self.logger.addHandler(logging.NullHandler())

        self.model = None
        if autosetup:
            try:
                self.load()
            except:
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

    def iterable_sentence_words(self):
        """Yield list of sentence words.

        Raises
        ------
        err : NotImplementedError
            Always raised as a derived class should define it.

        """
        raise NotImplementedError("Define this in derived class")

    def load(self, filename=DOC2VEC_FILENAME):
        """Load model from pickle file.

        This function is unsafe. Do not load unsafe files.

        Parameters
        ----------
        filename : str
            Filename of pickle file.

        """
        full_filename = self.full_filename(filename)
        self.logger.info('Loading doc2vec model from {}'.format(
            full_filename))
        self.model = gensim.models.Doc2Vec.load(full_filename)

    def save(self, filename=DOC2VEC_FILENAME):
        """Save model to pickle file.

        The Gensim load file is used which can also compress the file.

        Parameters
        ----------
        filename : str, optional
            Filename.

        """
        full_filename = self.full_filename(filename)
        self.model.save(full_filename)

    def train(self, size=100, window=8, min_count=5, workers=4):
        """Train Gensim Doc2Vec model.

        Parameters
        ----------
        size : int, optional
            Dimension of the word2vec space.

        """
        tagged_documents = self.iterable_tagged_documents()
        self.model = gensim.models.Doc2Vec(
            tagged_documents, size=size, window=window, min_count=min_count,
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
        >>> d2v = Doc2Vec()
        >>> d2v.doesnt_match(['svend', 'stol', 'ole', 'anders'])
        'stol'

        """
        return self.model.doesnt_match(words)

    def most_similar(self, positive=[], negative=[], topn=10,
                     restrict_vocab=None, indexer=None):
        """Return most similar words.

        This method will forward the similarity search to the `most_similar`
        method in the Doc2Vec class in Gensim. The input parameters and
        returned result are the same.

        Parameters
        ----------
        positive : list of str
            List of strings with words to include for similarity search.
        negative : list of str
            List of strings with words to discount.
        topn : int
            Number of words to return.
        restrict_vocab : int, optional
            The maximum size of the vocabulary. This is forwarded to the Gensim
            `restrict_vocab` parameter.

        Returns
        -------
        words : list of tuples
            List of 2-tuples with word and similarity.

        Examples
        --------
        >>> d2v = Doc2Vec()
        >>> words = w2v.most_similar('studieretning')
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


class FastText(object):
    """FastText abstract class."""

    def __init__(self, autosetup=True):
        """Set up logger."""
        self.logger = logging.getLogger(__name__ + '.FastText')
        self.logger.addHandler(logging.NullHandler())

        self.model = None
        if autosetup:
            self.logger.info('Autosetup')
            try:
                self.load()
            except ValueError:
                # The file is probably not there
                self.logger.info('Loading fasttext model failed. Training')
                self.train()

        # Projection matrix used for similarity computation
        self._normalized_matrix = None

    def data_directory(self):
        """Return data directory.

        Raises
        ------
        err : NotImplementedError
            Always raised as a derived class should define it.

        """
        raise NotImplementedError('Define this in derived class')

    def full_filename(self, filename):
        """Return filename with full filename path.

        Parameters
        ----------
        filename : str
            Filename. If the filename has no extension then the

        """
        if sep in filename:
            return filename
        else:
            return join(self.data_directory(), filename)

    def load(self, filename=None, model_type='skipgram'):
        """Load model from pickle file.

        This function is unsafe. Do not load unsafe files.

        Parameters
        ----------
        filename : str
            Filename of pickle file.
        model_type : skipgram, cbow, optional
            Type of fastText model [default: skipgram].

        """
        if filename is None:
            if model_type == 'skipgram':
                filename = FAST_TEXT_SKIPGRAM_MODEL_FILENAME
            elif model_type == 'cbow':
                filename = FAST_TEXT_CBOW_MODEL_FILENAME
            filename = filename + '.bin'

        full_filename = self.full_filename(filename)
        self.logger.info('Trying to load fastText model from {}'.format(
            full_filename))
        self.model = fasttext.load_model(full_filename, encoding='utf-8')

        # Invalidating cached computation
        self._normalized_matrix = None

    def make_data_directory(self):
        """Make data directory.

        Raises
        ------
        err : NotImplementedError
            Always raised as a derived class should define it.

        """
        raise NotImplementedError('Define this in derived class')

    def most_similar(self, word, top_n=10):
        """Return most similar words.

        Parameters
        ----------
        word : str
            Word to compare to trained model.
        top_n : int, optional
            Number of words to return [default: 10].

        Returns
        -------
        words : list of tuples
            List of 2-tuples with word and similarity.

        """
        word_vector = self.word_vector(word)

        model_words = list(self.model.words)

        if self._normalized_matrix is None:
            self.logger.info('Computing normalized matrix')
            self._normalized_matrix = zeros(
                (len(model_words), self.model.dim))
            for n, model_word in enumerate(model_words):
                self._normalized_matrix[n, :] = self.word_vector(model_word)
            self._normalized_matrix /= sqrt(
                (self._normalized_matrix ** 2).sum(-1))[..., newaxis]

        self.logger.debug('Searching over {} words'.format(len(model_words)))
        similarities = dot(self._normalized_matrix, word_vector)

        self.logger.debug('Sorting similarities')
        indices = argsort(-similarities)
        words_and_similarities = [
            (model_words[indices[n]], similarities[indices[n]])
            for n in range(top_n)]
        return words_and_similarities

    def train(self, input_filename=SENTENCES_FILENAME,
              model_filename=None,
              model_type='skipgram'):
        """Train a fasttext model.

        Parameters
        ----------
        input_filename : str, optional
            Filename for input file with sentences.
        model_filename : str, optional
            Filename for model output.
        model_type : skipgram or cbow, optional
            Model type.

        """
        if model_filename is None:
            if model_type == 'skipgram':
                model_filename = FAST_TEXT_SKIPGRAM_MODEL_FILENAME
            elif model_type == 'cbow':
                model_filename = FAST_TEXT_CBOW_MODEL_FILENAME

        full_model_filename = self.full_filename(model_filename)
        full_input_filename = self.full_filename(input_filename)

        if model_type == 'skipgram':
            self.logger.info(
                'Training fasttext skipgram model on {} to {}'.format(
                    full_input_filename, full_model_filename))
            self.model = fasttext.skipgram(
                full_input_filename, full_model_filename)
        elif model_type == 'cbow':
            self.logger.info(
                'Training fasttext cbow model on {} to {}'.format(
                    full_input_filename, full_model_filename))
            self.model = fasttext.cbow(
                full_input_filename, full_model_filename)
        else:
            raise ValueError('Wrong argument to model_type')

        # Invalidate computed normalized matrix
        self._normalized_matrix = None

    def word_vector(self, word):
        """Return feature vector for word.

        Parameters
        ----------
        word : str
            Word.

        Returns
        -------
        vector : np.array
            Array will values from FastText. If the word is not in the
            vocabulary, then a zero vector is returned.

        """
        return array(self.model[word])


class Word2Vec(object):
    """Gensim Word2vec abstract class.

    Parameters
    ----------
    autosetup : bool, optional
        Determines whether the Word2Vec model should be autoloaded.

    Notes
    -----
    Trained models can be saved and loaded via the `save` and `load` methods.

    """

    def __init__(self, autosetup=True):
        """Set up model."""
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
        """Make data directory."""
        make_data_directory(self.data_directory())

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

        """
        return self.model.doesnt_match(words)

    def most_similar(self, positive=[], negative=[], top_n=10,
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
        top_n : int
            Number of words to return
        restrict_vocab : int, optional
            The maximum size of the vocabulary. This is forwarded to the Gensim
            `restrict_vocab` parameter.

        Returns
        -------
        words : list of tuples
            List of 2-tuples with word and similarity.

        """
        return self.model.most_similar(
            positive, negative, top_n, restrict_vocab, indexer)

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
    print('dasem.models.FastText')
    print('dasem.models.Word2Vec')


if __name__ == '__main__':
    main()
