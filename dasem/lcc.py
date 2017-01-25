"""lcc - Leipzig Corpora Collection.

Usage:
  dasem.lcc data-directory
  dasem.lcc download
  dasem.lcc download-file <file>
  dasem.lcc get-sentence-words [options]
  dasem.lcc get-sentence-words-from-file [options] <file>
  dasem.lcc get-sentences [options]
  dasem.lcc get-sentences-from-file <file>
  dasem.lcc train-and-save-word2vec [options]

Options:
  --debug             Debug messages
  -h --help           Help message
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  --separator=<sep>   Separator [default: |]
  --verbose           Verbose messages

Examples:
  $ python -m dasem.lcc download-file dan-dk_web_2014_10K.tar.gz

  $ python -m dasem.lcc get-sentences | head -n30 | tail -3 | cut -f1-8 -d' '
  1. Hvordan vasker jeg bilen uden at lakken
  1 produkter produkt Kr 99,00 (tom) Du har
  20-06-2008 Kun nogenlunde amerikansk film om gambling. 13-06-2008

References:
  http://corpora2.informatik.uni-leipzig.de/download.html

"""

from __future__ import absolute_import, print_function

import errno

import logging

import os
from os import write
from os.path import isfile, join, split, splitext

import requests

from shutil import copyfileobj

from six import b, u

import tarfile

import nltk
from nltk.stem.snowball import DanishStemmer
from nltk.tokenize import WordPunctTokenizer

from .config import data_directory
from .utils import make_data_directory
from . import models


BASE_URL = 'http://corpora2.informatik.uni-leipzig.de/downloads/'

FILENAMES = [
    'dan-dk_web_2014_10K.tar.gz',
    'dan-dk_web_2014_1M.tar.gz',
    'dan_news_2007_1M-text.tar.gz',
    'dan_newscrawl_2011_1M-text.tar.gz',
]


class LCCFile(object):
    """Leipzig Corpora Collection file interface.

    Parameters
    ----------
    filename : str
        Filename for the .tar.gz file.

    Attributes
    ----------
    stemmer : object with stem method
        Object with stem method corresponding to
        nltk.stem.snowball.DanishStemmer.
    word_tokenizer : object with tokenize method
        Object with tokenize method, corresponding to nltk.WordPunctTokenizer.

    """

    def __init__(self, filename):
        """Setup filename."""
        self.filename = filename
        self.word_tokenizer = WordPunctTokenizer()
        self.stemmer = DanishStemmer()

    def iter_sentences(self):
        """Yield sentences.

        Reads from the *-sentences.txt' file.

        Yields
        ------
        sentence : str
            Sentences as Unicode strings.

        """
        _, filename_tail = split(self.filename)
        filename_base, _ = splitext(splitext(filename_tail)[0])
        with tarfile.open(self.filename, "r:gz") as tar:
            sentence_filename = join(filename_base, filename_base +
                                     '-sentences.txt')
            try:
                fid = tar.extractfile(sentence_filename)
            except KeyError:
                # Try another name
                sentence_filename = filename_base[:-5] + '-sentences.txt'
                fid = tar.extractfile(sentence_filename)

            for line in fid:
                yield line.decode('utf-8').split('\t')[1].strip()

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
        for n, sentence in enumerate(self.iter_sentences()):
            words = self.word_tokenizer.tokenize(sentence)
            if lower:
                words = [word.lower() for word in words]
            if stem:
                words = [self.stemmer.stem(word) for word in words]

            yield words
            if n > 10: break
                
class LCC(object):
    """Leipzig Corpora Collection interface.

    References
    ----------
    - http://corpora2.informatik.uni-leipzig.de/download.html
    - Quasthoff, U.; M. Richter; C. Biemann: Corpus Portal for Search in
      Monolingual Corpora, Proceedings of the fifth international conference
      on Language Resources and Evaluation, LREC 2006, Genoa, pp. 1799-1802

    """

    def data_directory(self):
        """Return diretory where data should be.

        Returns
        -------
        dir : str
            Directory.

        """
        dir = join(data_directory(), 'lcc')
        return dir

    def download_file(self, filename, redownload=False):
        """Download a file.

        Parameters
        ----------
        filename : str
            Filename without server or path information.

        Examples
        --------
        filename = 'dan-dk_web_2014_10K.tar.gz'

        """
        local_filename = join(self.data_directory(), filename)
        if not redownload and isfile(local_filename):
            return

        self.make_data_directory()
        url = BASE_URL + filename
        response = requests.get(url, stream=True)
        with open(local_filename, 'wb') as fid:
            copyfileobj(response.raw, fid)

    def download(self):
        """Download data."""
        self.make_data_directory()
        for filename in FILENAMES:
            self.download_file(filename)

    def iter_sentences(self):
        """Iterate over all sentences.

        Yields
        ------
        sentence : str
            Sentences as string from all files.

        """
        self.download()
        for filename in FILENAMES:
            full_filename = join(self.data_directory(), filename)
            lcc_file = LCCFile(full_filename)
            for sentence in lcc_file.iter_sentences():
                yield sentence

    def iter_sentence_words(self, lower=True, stem=False):
        """Iterate over all sentences return a word list.

        Parameters
        ----------
        lower : bool, default True
            Lower case the words.
        stem : bool, default False
            Apply word stemming. DanishStemmer from nltk is used.

        Yields
        ------
        word_list : list of str
            List of string with words from sentences.

        """
        for filename in FILENAMES:
            full_filename = join(self.data_directory(), filename)
            lcc_file = LCCFile(full_filename)
            for word_list in lcc_file.iter_sentence_words():
                if lower:
                    word_list = [word.lower() for word in word_list]
                if stem:
                    word_list = [self.stemmer.stem(word) for word in word_lits]
                yield word_list

                
    def make_data_directory(self):
        """Make data directory for LCC."""
        make_data_directory(data_directory(), 'lcc')


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
    
    def __iter__(self):
        """Restart and return iterable."""
        lcc = LCC()
        sentences = lcc.iter_sentence_words(
            lower=self.lower, stem=self.stem)
        return sentences
        

class Word2Vec(models.Word2Vec):

    def data_directory(self):
        """Return data directory.

        Returns
        -------
        dir : str
            Directory for data.

        """
        dir = join(data_directory(), 'lcc')
        return dir

    def iterable_sentence_words(self, lower=True, stem=False):
        """Returns iterable for sentence words.

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


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)
    logging_level = logging.WARN
    if arguments['--debug']:
        logging_level = logging.DEBUG
    elif arguments['--verbose']:
        logging_level = logging.INFO

    logger = logging.getLogger('dasem.gutenberg')
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

    if arguments['data-directory']:
        lcc = LCC()
        print(lcc.data_directory())

    elif arguments['download']:
        lcc = LCC()
        lcc.download()

    elif arguments['download-file']:
        filename = arguments['<file>']
        lcc = LCC()
        lcc.download_file(filename)

    elif arguments['get-sentence-words']:
        filename = arguments['<file>']
        separator = u(arguments['--separator'])
        lcc = LCC()
        try:
            for word_list in lcc.iter_sentence_words():
                write(output_file,
                      separator.join(word_list).encode(encoding) + b('\n'))
        except Exception as err:
            if err.errno != errno.EPIPE:
                raise
            else:
                # if piped to the head command
                pass

    elif arguments['get-sentence-words-from-file']:
        filename = arguments['<file>']
        separator = u(arguments['--separator'])
        lcc_file = LCCFile(filename)
        try:
            for word_list in lcc_file.iter_sentence_words():
                write(output_file,
                      separator.join(word_list).encode(encoding) + b('\n'))
        except Exception as err:
            if err.errno != errno.EPIPE:
                raise
            else:
                # if piped to the head command
                pass

    elif arguments['get-sentences']:
        lcc = LCC()
        try:
            for sentence in lcc.iter_sentences():
                write(output_file, sentence.encode(encoding) + b('\n'))
        except Exception as err:
            if err.errno != errno.EPIPE:
                raise
            else:
                # if piped to the head command
                pass

    elif arguments['get-sentences-from-file']:
        filename = arguments['<file>']
        lcc_file = LCCFile(filename)
        for sentence in lcc_file.iter_sentences():
            print(sentence)

    elif arguments['train-and-save-word2vec']:
        print(1)
        word2vec = Word2Vec(autosetup=False)
        print(2)
        word2vec.train()
        logger.info('Saving word2vec model')
        word2vec.save()

            
if __name__ == "__main__":
    main()