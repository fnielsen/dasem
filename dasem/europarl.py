"""europarl.

Usage:
  dasem.europarl download [options]
  dasem.europarl get-all-sentence-words [options]
  dasem.europarl get-all-sentences [options]
  dasem.europarl get-all-tokenized-sentences [options]

Options:
  --debug             Debug messages.
  -h --help           Help message
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  --separator=<sep>   Separator [default: |]
  --verbose           Verbose messages.

Description:
  This module handles the interface to the Danish part of the Europarl corpus
  available at http://www.statmt.org/europarl/

  It will work from the Danish-English parallel corpus available from:

      http://www.statmt.org/europarl/v7/da-en.tgz

Example:
  $ python -m dasem.europarl download --debug

"""


from __future__ import absolute_import, division, print_function

import logging

import os
from os import write
from os.path import isfile, join

from shutil import copyfileobj

import signal

import tarfile

import requests

from six import b, u

from nltk.stem.snowball import DanishStemmer
from nltk.tokenize import WordPunctTokenizer

from .config import data_directory
from .corpus import Corpus
from .utils import make_data_directory


TAR_GZ_FILENAME = 'da-en.tar.gz'
TGZ_PARALLEL_CORPUS_FILENAME = "da-en.tgz"

DANISH_FILENAME = 'europarl-v7.da-en.da'

TGZ_PARALLEL_CORPUS_URL = "http://www.statmt.org/europarl/v7/da-en.tgz"


class Europarl(Corpus):
    """Europarl corpus.

    Examples
    --------
    >>> europarl = Europarl()
    >>> sentence = next(europarl.iter_tokenized_sentences())
    >>> "sessionen" in sentence.split()
    True

    """

    def __init__(self, danish_filename=DANISH_FILENAME,
                 tar_gz_filename=TGZ_PARALLEL_CORPUS_FILENAME):
        """Set up filename.

        Parameters
        ----------
        danish_filename : str
            Filename for '.da' file in the tar.gz file.
        tar_gz_filename : str
            Filename for tar.gz or tgz file with Danish/English.

        """
        self.logger = logging.getLogger(__name__ + '.Europarl')
        self.logger.addHandler(logging.NullHandler())

        self.tar_gz_filename = tar_gz_filename
        self.danish_filename = danish_filename

        self.word_tokenizer = WordPunctTokenizer()
        self.stemmer = DanishStemmer()

    def data_directory(self):
        """Return diretory where data should be.

        Returns
        -------
        directory : str
            Directory.

        """
        directory = join(data_directory(), 'europarl')
        return directory

    def download(self, redownload=False):
        """Download corpus."""
        filename = TGZ_PARALLEL_CORPUS_FILENAME
        local_filename = join(self.data_directory(), filename)
        if not redownload and isfile(local_filename):
            message = 'Not downloading as corpus already download to {}'
            self.logger.debug(message.format(local_filename))
            return

        self.make_data_directory()
        url = TGZ_PARALLEL_CORPUS_URL
        self.logger.info('Downloading {} to {}'.format(url, local_filename))
        response = requests.get(url, stream=True)
        with open(local_filename, 'wb') as fid:
            copyfileobj(response.raw, fid)
        self.logger.debug('Corpus downloaded'.format())

    def iter_sentences(self):
        """Yield sentences.

        Yields
        ------
        sentence : str
            Sentences as Unicode strings.

        """
        full_tar_gz_filename = join(self.data_directory(),
                                    self.tar_gz_filename)
        with tarfile.open(full_tar_gz_filename, "r:gz") as tar:
            fid = tar.extractfile(self.danish_filename)
            for line in fid:
                yield line.decode('utf-8').strip()

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

    def make_data_directory(self):
        """Make data directory for Europarl."""
        make_data_directory(self.data_directory())


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
    separator = u(arguments['--separator'])

    # Ignore broken pipe errors
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if arguments['download']:
        europarl = Europarl()
        europarl.download()

    elif arguments['get-all-sentence-words']:
        europarl = Europarl()
        for word_list in europarl.iter_sentence_words():
            write(output_file,
                  separator.join(word_list).encode(output_encoding) + b('\n'))

    elif arguments['get-all-sentences']:
        europarl = Europarl()
        for sentence in europarl.iter_sentences():
            write(output_file, sentence.encode(output_encoding) + b('\n'))

    elif arguments['get-all-tokenized-sentences']:
        europarl = Europarl()
        for sentence in europarl.iter_tokenized_sentences():
            write(output_file, sentence.encode(output_encoding) + b('\n'))

    else:
        assert False


if __name__ == '__main__':
    main()
