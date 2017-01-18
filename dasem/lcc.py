"""lcc - Leipzig Corpora Collection.

Usage:
  dasem.lcc data-directory
  dasem.lcc download
  dasem.lcc download-file <file>
  dasem.lcc get-sentences [options]
  dasem.lcc get-sentences-from-file <file>

Options:
  -h --help         Help message
  --oe=encoding     Output encoding [default: utf-8]
  -o --output=file  Output filename, default output to stdout

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

import os
from os import write
from os.path import isfile, join, split, splitext

import requests

from shutil import copyfileobj

from six import b

import tarfile

from .config import data_directory
from .utils import make_data_directory


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

    """

    def __init__(self, filename):
        """Setup filename."""
        self.filename = filename

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

    def make_data_directory(self):
        """Make data directory for LCC."""
        make_data_directory(data_directory(), 'lcc')


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)
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


if __name__ == "__main__":
    main()
