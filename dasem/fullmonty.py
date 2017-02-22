"""fullmonty.

Usage:
  dasem.fullmonty get-all-sentences [options]
  dasem.fullmonty fasttext-most-similar [options] <word>
  dasem.fullmonty most-similar [options] <word>
  dasem.fullmonty train-and-save-fasttext [options]
  dasem.fullmonty train-and-save-word2vec [options]

Options:
  --debug             Debug messages.
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
  $ python -m dasem.fullmonty most-similar -n 1 mand
  kvinde

"""


from __future__ import absolute_import, division, print_function

from itertools import chain

import logging

import os
from os import write
from os.path import join

import signal

from six import b, text_type, u

from . import models
from .config import data_directory
from .dannet import Dannet
from .europarl import Europarl
from .gutenberg import Gutenberg
from .lcc import LCC
from .utils import make_data_directory


class Fullmonty(object):
    """All corpora."""

    def __init__(self):
        """Setup objects for corpora."""
        self.dannet = Dannet()
        self.gutenberg = Gutenberg()
        self.lcc = LCC()
        self.europarl = Europarl()

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
        """Setup options."""
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


class Word2Vec(models.Word2Vec):
    """Word2Vec model with automated load of all corpora."""

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

    if arguments['fasttext-most-similar']:
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
