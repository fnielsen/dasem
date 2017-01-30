"""europarl.

Usage:
  dasem.europarl get-all-sentence-words [options]
  dasem.europarl get-all-sentences [options]

Options:
  --debug             Debug messages.
  -h --help           Help message
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  --separator=<sep>   Separator [default: |]
  --verbose           Verbose messages.

"""


from __future__ import absolute_import, division, print_function

import errno

import logging

import os
from os import write
from os.path import join

# http://stackoverflow.com/questions/34718208/
import socket

import tarfile

from six import b, u

from nltk.stem.snowball import DanishStemmer
from nltk.tokenize import WordPunctTokenizer

from .config import data_directory


TAR_GZ_FILENAME = 'da-en.tar.gz'

DANISH_FILENAME = 'europarl-v7.da-en.da'


class Europarl(object):
    """Europarl corpus."""

    def __init__(self, danish_filename=DANISH_FILENAME,
                 tar_gz_filename=TAR_GZ_FILENAME):
        """Setup filename.

        Parameters
        ----------
        danish_filename : str
            Filename for '.da' file in the tar.gz file.
        tar_gz_filename : str
            Filename for tar.gz file with Danish/English.

        """
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
    encoding = arguments['--oe']
    separator = u(arguments['--separator'])

    if arguments['get-all-sentence-words']:
        europarl = Europarl()
        try:
            for word_list in europarl.iter_sentence_words():
                write(output_file,
                      separator.join(word_list).encode(encoding) + b('\n'))
        except socket.error as err:
            if err.errno != errno.EPIPE:
                raise
            else:
                # if piped to the head command
                pass

    elif arguments['get-all-sentences']:
        europarl = Europarl()
        try:
            for sentence in europarl.iter_sentences():
                write(output_file, sentence.encode(encoding) + b('\n'))
        except socket.error as err:
            if err.errno != errno.EPIPE:
                raise
            else:
                # if piped to the head command
                pass

    else:
        assert False


if __name__ == '__main__':
    main()
