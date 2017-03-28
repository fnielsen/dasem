"""dasem.

Usage:
   dasem decompound [options] <text>
   dasem most-similar [options] <word>

Options:
  --debug             Debug messages.
  --ie=encoding       Input encoding [default: utf-8]
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  -h --help
  --verbose           Verbose messages.

"""


from __future__ import absolute_import, division, print_function

import errno

import logging

import os
from os import write

import socket

from six import b, text_type

from .fullmonty import FastText, Word2Vec
from .text import Decompounder


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

    if arguments['decompound']:
        text = arguments['<text>']
        if not isinstance(text, text_type):
            text = text.decode(input_encoding)
        text = text.lower()

        decompounder = Decompounder()
        decompounded = decompounder.decompound_text(text)
        write(output_file, decompounded.encode(output_encoding) + b('\n'))

    elif arguments['most-similar']:
        word = arguments['<word>']
        if not isinstance(word, text_type):
            word = word.decode(input_encoding)
        word = word.lower()
        word2vec = Word2Vec()
        try:
            words_and_similarity = word2vec.most_similar(word)
        except:
            fast_text = FastText()
            words_and_similarity = fast_text.most_similar(word)
        try:
            for word, _ in words_and_similarity:
                write(output_file, word.encode(output_encoding) + b('\n'))
        except socket.error as err:
            if err.errno != errno.EPIPE:
                raise
            else:
                # if piped to the head command
                pass


if __name__ == '__main__':
    main()
