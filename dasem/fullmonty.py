"""fullmonty.

Usage:
  dasem.fullmonty download [options]
  dasem.fullmonty get-all-sentences [options]
  dasem.fullmonty get-all-tokenized-sentences [options]
  dasem.fullmonty fasttext-most-similar [options] <word>
  dasem.fullmonty most-similar [options] <word>
  dasem.fullmonty train-and-save-fasttext [options]
  dasem.fullmonty train-and-save-word2vec [options]

Options:
  --debug             Debug messages.
  -i=<filename>       Input filename
  --ie=encoding       Input encoding [default: utf-8]
  --oe=encoding       Output encoding [default: utf-8]
  -n=<n> | --n=<n>    Number. For most-similar command, the top number
                      of items to return
  -o --output=<file>  Output filename, default output to stdout
  -h --help
  --verbose           Verbose messages.
  --with-scores       For the most-similar command, print the score
                      along with the words

Examples:
  $ python -m dasem.fullmonty download --verbose

  $ python -m dasem.fullmonty most-similar -n 1 mand
  kvinde

"""


from __future__ import absolute_import, division, print_function

import codecs

from collections import Counter

from itertools import chain

import logging

from math import log

import os
from os import write
from os.path import join, sep

import signal

from nltk.tokenize import WordPunctTokenizer

from six import b, text_type, u

from . import models
from .config import data_directory
from .corpus import Corpus
from .dannet import Dannet
from .europarl import Europarl
from .gutenberg import Gutenberg
from .lcc import LCC
from .utils import make_data_directory


TOKENIZED_SENTENCES_FILENAME = 'tokenized_sentences.txt'

WORD_COUNTS_FILENAME = 'word_counts.txt'


class DataDirectoryMixin(object):
    """Class to specify data directory.

    This class should have first inheritance, so that its `data_directory`
    method is calle before the abstract class.

    """

    def data_directory(self):
        """Return diretory where data should be.

        Returns
        -------
        directory : str
            Directory.

        """
        directory = join(data_directory(), 'fullmonty')
        return directory

    def full_filename(self, filename):
        """Prepend data directory path to filename.

        Parameters
        ----------
        filename : str
            Filename of local file.

        Returns
        -------
        full_filename : str
            Filename with full directory path information.

        """
        if sep in filename:
            return filename
        else:
            return join(self.data_directory(), filename)


class Fullmonty(Corpus):
    """All corpora.

    The corpora included in the Fullmonty aggregated corpora ae
    Dannet, Gutenberg, LCC and Europarl.

    """

    def __init__(self):
        """Set up objects for logger and corpora."""
        super(self.__class__, self).__init__()

        self.logger = logging.getLogger(__name__ + '.Fullmonty')
        self.logger.addHandler(logging.NullHandler())

        self.dannet = Dannet()
        self.gutenberg = Gutenberg()
        self.lcc = LCC()
        self.europarl = Europarl()

    def download(self):
        """Download all corpora."""
        self.logger.info('Downloading all corpora')
        self.dannet.download()
        self.gutenberg.download()
        self.lcc.download()
        self.europarl.download()
        self.logger.debug('All corpora downloaded')

    def iter_sentences(self):
        """Iterate over sentences from all corpora.

        Yields
        ------
        sentences : str
            Sentence from corpora as string.

        """
        dannet_sentences = self.dannet.iter_sentences()
        europarl_sentences = self.europarl.iter_sentences()
        gutenberg_sentences = self.gutenberg.iter_sentences()
        lcc_sentences = self.lcc.iter_sentences()

        sentences = chain(dannet_sentences, europarl_sentences,
                          gutenberg_sentences, lcc_sentences)
        return sentences


class SentenceWordsIterable(object):
    """Iterable for words in a sentence.

    Parameters
    ----------
    lower : bool, default True
        Lower case the words.
    stem : bool, default False
        Apply word stemming. DanishStemmer from nltk is used.

    """

    def __init__(self, lower=True, stem=False):
        """Set up options."""
        self.lower = lower
        self.stem = stem

        self.dannet = Dannet()
        self.europarl = Europarl()
        self.gutenberg = Gutenberg()
        self.lcc = LCC()

    def __iter__(self):
        """Restart and return iterable."""
        dannet_sentence_words = self.dannet.iter_sentence_words(
            lower=self.lower, stem=self.stem)

        europarl_sentence_words = self.europarl.iter_sentence_words(
            lower=self.lower, stem=self.stem)

        gutenberg_sentence_words = self.gutenberg.iter_sentence_words(
            lower=self.lower, stem=self.stem)

        lcc_sentence_words = self.lcc.iter_sentence_words(
            lower=self.lower, stem=self.stem)

        sentence_words = chain(dannet_sentence_words, europarl_sentence_words,
                               gutenberg_sentence_words, lcc_sentence_words)
        return sentence_words


class FastText(models.FastText):
    """FastText model for fullmonty dataset."""

    def data_directory(self):
        """Return data directory.

        Returns
        -------
        directory : str
            Directory for data.

        """
        directory = join(data_directory(), 'fullmonty')
        return directory


class TokenizedSentences(DataDirectoryMixin):
    """Interface to tokenized sentences."""

    def __init__(self, filename=TOKENIZED_SENTENCES_FILENAME):
        """Initialize logger and filename.

        Parameters
        ----------
        filename : str
            filename with tokenized sentences.

        """
        self.logger = logging.getLogger(__name__ + '.TokenizedSentences')
        self.logger.addHandler(logging.NullHandler())

        self.filename = self.full_filename(filename)

    def count_words(self):
        """Count words.

        Returns
        -------
        word_counts : dict
            Counts for individual words in a dictionary.

        """
        words = []
        with codecs.open(self.filename, encoding='utf-8') as fid:
            for line in fid:
                words.extend(line.split())
        word_counts = Counter(words)
        return word_counts


class Word2Vec(models.Word2Vec):
    """Word2Vec model with automated load of all corpora.

    Examples
    --------
    >>> w2v = Word2Vec()
    >>> w2v.doesnt_match(['svend', 'stol', 'ole', 'anders'])
    'stol'

    >>> words = w2v.most_similar('studie')
    >>> len(words)
    10

    """

    def data_directory(self):
        """Return data directory.

        Returns
        -------
        directory : str
            Directory for data.

        """
        directory = join(data_directory(), 'fullmonty')
        return directory

    def iterable_sentence_words(self, lower=True, stem=False):
        """Return iterable for sentence words.

        Parameters
        ----------
        lower : bool, default True
            Lower case the words.
        stem : bool, default False
            Apply word stemming. DanishStemmer from nltk is used.

        Returns
        -------
        sentence_words : iterable
            Iterable over sentence words

        """
        sentence_words = SentenceWordsIterable(lower=lower, stem=stem)
        return sentence_words

    def make_data_directory(self):
        """Make data directory for fullmonty."""
        make_data_directory(data_directory(), 'fullmonty')


class WordCounts(DataDirectoryMixin):
    """Word counts.

    Parameters
    ----------
    filename : str, optional
        Filename of word counts file.

    """

    def __init__(self, filename=WORD_COUNTS_FILENAME):
        """Set up counts."""
        self.filename = self.full_filename(filename)

        # Set up counts
        try:
            self.load_counts()
        except IOError:
            self.setup_counts_from_tokenized_sentences()
            self.save_counts()

        self.word_tokenizer = WordPunctTokenizer()

    def load_counts(self):
        """Load word count data."""
        with codecs.open(self.filename, encoding='utf-8') as fid:
            self._word_counts = Counter(
                {word: int(count)
                 for count, word in (line.split() for line in fid)})
        self.setup_count_sum()

    def save_counts(self):
        """Save word counts data."""
        with codecs.open(self.filename, 'w', encoding='utf-8') as fid:
            for word, count in self._word_counts.items():
                fid.write(u('{} {}\n').format(count, word))

    def setup_count_sum(self):
        """Sum the word counts."""
        self._count_sum = sum(self._word_counts.values())

    def setup_counts_from_tokenized_sentences(self):
        """Count and set words in tokenized sentences."""
        tokenized_sentences = TokenizedSentences()
        self._word_counts = tokenized_sentences.count_words()
        self.setup_count_sum()

    def word_surprisal_bits(self, word):
        """Return surprisal for word in bits.

        Parameters
        ----------
        word : str
            Word as string.

        Returns
        -------
        bits : float
            surprisal as bits

        References
        ----------
        - https://en.wikipedia.org/wiki/Self-information

        Examples
        --------
        >>> word_counts = WordCounts()
        >>> word_counts.word_surprisal_bits('at') < 10
        True

        >>> word_counts.word_surprisal_bits('rekrutteringsrunde') > 10
        True

        """
        # Add a prior on 1 for all words
        bits = - log((self._word_counts[word] + 1) /
                     (self._count_sum + len(self._word_counts)), 2)
        return bits

    def extract_keywords(self, text, top_n=10):
        """Extract keywords from text.

        The method is based on surprisal.

        Parameters
        ----------
        text : str
            Text with words
        top_n : int, optional
            Number of words to return.

        Returns
        -------
        top_words : list of str
            List of string for words with highest surprisal.

        """
        words = list(set(word.lower()
                         for word in self.word_tokenizer.tokenize(text)))
        surprisals = [self.word_surprisal_bits(word) for word in words]

        indices = sorted(range(len(surprisals)),
                         key=lambda i: -surprisals[i])[:top_n]
        top_words = [words[i] for i in indices]
        return top_words


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    logging_level = logging.WARN
    if arguments['--debug']:
        logging_level = logging.DEBUG
    elif arguments['--verbose']:
        logging_level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(logging_level)
    logging_handler = logging.StreamHandler()
    logging_handler.setLevel(logging_level)
    logging_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging_handler.setFormatter(logging_formatter)
    logger.addHandler(logging_handler)

    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if arguments['--output']:
        output_filename = arguments['--output']
        output_file = os.open(output_filename, os.O_RDWR | os.O_CREAT)
    else:
        # stdout
        output_file = 1
    output_encoding = arguments['--oe']
    input_encoding = arguments['--ie']

    input_filename = arguments['-i']

    if arguments['download']:
        fullmonty = Fullmonty()
        fullmonty.download()

    elif arguments['fasttext-most-similar']:
        word = arguments['<word>']
        if not isinstance(word, text_type):
            word = word.decode(input_encoding)

        top_n = arguments['--n']
        if top_n is None:
            top_n = 10
        top_n = int(top_n)

        fast_text = FastText()
        for word, similarity in fast_text.most_similar(word, top_n=top_n):
            write(output_file, word.encode('utf-8') + b('\n'))

    elif arguments['get-all-sentences']:
        fullmonty = Fullmonty()
        for sentence in fullmonty.iter_sentences():
            write(output_file, sentence.encode(output_encoding) + b('\n'))

    elif arguments['get-all-tokenized-sentences']:
        fullmonty = Fullmonty()
        for sentence in fullmonty.iter_tokenized_sentences():
            write(output_file, sentence.encode(output_encoding) + b('\n'))

    elif arguments['most-similar']:

        word = arguments['<word>']
        if not isinstance(word, text_type):
            word = word.decode(input_encoding)
        word = word.lower()

        top_n = arguments['--n']
        if top_n is None:
            top_n = 10
        top_n = int(top_n)

        word2vec = Word2Vec()
        words_and_similarity = word2vec.most_similar(word, top_n=top_n)

        for word, score in words_and_similarity:
            if arguments['--with-scores']:
                format_spec = u("{0:+15.12f} {1}")
            else:
                format_spec = u("{1}")
            write(output_file, format_spec.format(
                score, word).encode(output_encoding) + b('\n'))

    elif arguments['train-and-save-fasttext']:
        fast_text = FastText(autosetup=False)
        if input_filename:
            fast_text.train(input_filename=input_filename)
        else:
            fast_text.train()

    elif arguments['train-and-save-word2vec']:
        word2vec = Word2Vec(autosetup=False)
        logger.info('Training word2vec model')
        word2vec.train()
        logger.info('Saving word2vec model')
        word2vec.save()

    else:
        assert False


if __name__ == "__main__":
    main()
