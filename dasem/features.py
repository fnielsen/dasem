"""features.

Usage:
  dasem.features lines-to-feature-matrix [options]

Options:
  --debug             Debug messages.
  -h --help           Help message
  -i --input=<file>   Input file
  --ie=encoding       Input encoding [default: utf-8]
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  --separator=<sep>   Separator [default: |]
  --verbose           Verbose messages.

"""


from __future__ import absolute_import, division, print_function

import codecs

import logging

import signal

import sys

from afinn import Afinn

from nltk import WordPunctTokenizer

import numpy as np

from sklearn.base import BaseEstimator


class FeatureExtractor(BaseEstimator):
    """Feature extractor for Danish texts."""

    def __init__(self):
        """Set up text processors."""
        self.afinn = Afinn(language='da')
        self.word_tokenizer = WordPunctTokenizer()

    def partial_fit(self, Y, y=None):
        """Fit model.

        This is a dummy function.

        """
        return self

    def fit(self, X, y=None):
        """Fit model.

        This is a dummy function.

        """
        return self

    @property
    def features_(self):
        """Set up features."""
        features = [
            'n_characters',
            'n_words',
            'n_unique_words',
            'afinn_sum_valence',
            'afinn_sum_arousal',
            'afinn_sum_ambiguity'
        ]
        return features

    def transform(self, raw_documents, y=None):
        """Transform documents to features.

        Parameters
        ----------
        raw_documents : iterable over str
            Iterable with corpus to be transformed.
        y : numpy.array
            Target (not used, dummy parameter).

        """
        X = []
        for n, document in enumerate(raw_documents):
            words = self.word_tokenizer.tokenize(document)
            unique_words = set(words)
            scores = self.afinn.scores(document)
            sum_valence = sum(scores)
            sum_arousal = np.sum(np.abs(scores))

            X.append([
                len(document),
                len(words),
                len(unique_words),
                sum_valence,
                sum_arousal,
                sum_arousal - abs(sum_valence)
            ])

        X = np.array(X)
        return X

    fit_transform = transform


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

    # Ignore broken pipe errors
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if arguments['--output']:
        output_filename = arguments['--output']
    else:
        output_filename = None
    if arguments['--input']:
        input_filename = arguments['--input']
    else:
        input_filename = None

    input_encoding = arguments['--ie']

    if arguments['lines-to-feature-matrix']:
        extractor = FeatureExtractor()

        version = int(sys.version.split('.')[0])

        if input_filename:
            input_file = codecs.open(input_filename, encoding=input_encoding)
        else:
            if version == 2:
                input_file = codecs.getreader(input_encoding)(sys.stdin)
            elif version == 3:
                input_file = codecs.getreader(input_encoding)(sys.stdin.buffer)
            else:
                assert False
            input_filename = 'STDIN'
        logger.info('Reading text from {}'.format(input_filename))
        X = extractor.transform(input_file)

        header = "," + ",".join(extractor.features_)
        if output_filename:
            logger.info('Writing data to {}'.format(output_filename))
            np.savetxt(output_filename, X, header=header)
        else:
            logger.info('Writing data to STDOUT'.format(output_filename))
            if version == 2:
                np.savetxt(sys.stdout, X, header=header)
            elif version == 3:
                np.savetxt(sys.stdout.buffer, X, header=header)
            else:
                assert False

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
