"""fullmonty.

Usage:
  dasem.fullmonty get-all-sentences [options]
  dasem.fullmonty most-similar [options] <word>
  dasem.fullmonty train-and-save-word2vec [options]

Options:
  --debug             Debug messages.
  --ie=encoding     Input encoding [default: utf-8]
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  -h --help
  --verbose           Verbose messages.

"""


from __future__ import absolute_import, division, print_function

import errno

from itertools import chain

import logging

import os
from os import write
from os.path import join

from six import b, text_type

from . import models
from .config import data_directory
from .dannet import Dannet
from .gutenberg import Gutenberg
from .lcc import LCC
from .utils import make_data_directory


class Fullmonty(object):

    def __init__(self):
        """Setup objects for corpora."""
        self.dannet = Dannet()
        self.gutenberg = Gutenberg()
        self.lcc = LCC()

    def iter_sentences(self):
        """Iterate over sentences from all corpora.

        Yields
        ------
        sentences : str
            Sentence from corpora as string.

        """
        dannet_sentences = self.dannet.iter_sentences()
        gutenberg_sentences = self.gutenberg.iter_sentences()
        lcc_sentences = self.lcc.iter_sentences()

        sentences = chain(dannet_sentences, gutenberg_sentences)
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
        self.gutenberg = Gutenberg()
        self.lcc = LCC()

    def __iter__(self):
        """Restart and return iterable."""
        dannet_sentence_words = self.dannet.iter_sentence_words(
            lower=self.lower, stem=self.stem)
        
        gutenberg_sentence_words = self.gutenberg.iter_sentence_words(
            lower=self.lower, stem=self.stem)

        lcc_sentence_words = self.lcc.iter_sentence_words(
            lower=self.lower, stem=self.stem)

        sentence_words = chain(dannet_sentence_words, gutenberg_sentence_words,
                               lcc_sentence_words)
        return sentence_words


class Word2Vec(models.Word2Vec):
    """Word2Vec model with automated load of all corpora."""

    def data_directory(self):
        """Return data directory.

        Returns
        -------
        dir : str
            Directory for data.

        """
        dir = join(data_directory(), 'fullmonty')
        return dir

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
        """Make data directory for LCC."""
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

    if arguments['--output']:
        output_filename = arguments['--output']
        output_file = os.open(output_filename, os.O_RDWR | os.O_CREAT)
    else:
        # stdout
        output_file = 1
    output_encoding = arguments['--oe']
    input_encoding = arguments['--ie']
    
    if arguments['get-all-sentences']:
        fullmonty = Fullmonty()
        try:
            for sentence in fullmonty.iter_sentences():
                write(output_file, sentence.encode(output_encoding) + b('\n'))
        except Exception as err:
            print(err)
            if err.errno != errno.EPIPE:
                raise
            else:
                # if piped to the head command
                pass

    elif arguments['most-similar']:
        word = arguments['<word>']
        if not isinstance(word, text_type):
            word = word.decode(input_encoding)
        word = word.lower()
        word2vec = Word2Vec()
        words_and_similarity = word2vec.most_similar(word)
        for word, similarity in words_and_similarity:
            write(output_file, word.encode(output_encoding) + b('\n'))

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
