"""eparole - Interface to DSL's ePAROLE dataset.

Usage:
  dasem [options] word-to-lemma <word>

Options:
  --debug             Debug messages.
  -h --help           Help message
  --ie=encoding     Input encoding [default: utf-8]
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  --verbose           Verbose messages.

References:
  http://korpus.dsl.dk/resources.html

"""

from collections import Counter, defaultdict

import csv

import logging

import os
from os import write
from os.path import join, sep

from zipfile import ZipFile

from pandas import DataFrame, read_csv

from six import b, text_type

import requests

from .config import data_directory
from .utils import make_data_directory


ZIP_URL = 'http://korpus.dsl.dk/resources/corpora/ePAROLE.zip'

ZIP_FILENAME = 'ePAROLE.zip'

CSV_FILENAME = 'ePAROLE.csv'


class EParole(object):
    """Interface to ePAROLE dataset from DSL."""

    def __init__(self, password=None):
        """Set up variables.

        Parameters
        ----------
        password : str
            Password to encrypt zip file. This should only necessary the
            very first time the class in instanced.
        logging_level : logging.DEBUG, logging.INFO, ..., optional
            Logging level.

        """
        self.logger = logging.getLogger(__name__ + '.EParole')
        self.logger.addHandler(logging.NullHandler())

        self.word_to_lemmas_map = defaultdict(list)

        self.setup(password=password)

    def setup(self, password=None):
        """Set up data directory and data file.

        Parameters
        ----------
        password : str
            Password to the encrypted zip file.

        """
        full_dirname = self.full_filename()
        make_data_directory(full_dirname)

        # Dummy read to setup file
        self.read(password=password)

    def full_filename(self, filename=None):
        """Prepend data directory path to filename.

        Parameters
        ----------
        filename : str, optional
            Filename of local Dannet file. If empty the data directory is
            returned.

        Returns
        -------
        full_filename : str
            Filename with full directory path information.

        """
        if filename is None:
            return join(data_directory(), 'eparole')
        elif sep in filename:
            return filename
        else:
            return join(data_directory(), 'eparole', filename)

    def download(self, url=ZIP_URL, filename=ZIP_FILENAME):
        """Download zip file."""
        full_filename = self.full_filename(filename)

        response = requests.get(url, stream=True)
        with open(full_filename, 'w') as f:
            for data in response.iter_content():
                f.write(data)

    def read(self, filename=CSV_FILENAME, password=None,
             zip_filename=ZIP_FILENAME):
        """Read data from extract file.

        Parameters
        ----------
        filename : str, optional
            Filename for a comma-separated file with the extracted data.
        password : str, optional
            Password for encrypted zip archive.
        zip_filename : str, optional
            Filename for zip archive.

        """
        full_csv_filename = self.full_filename(filename)
        self.logger.info('Trying to read data from {}'.format(
            full_csv_filename))
        try:
            df = read_csv(full_csv_filename, encoding='utf-8', index_col=0)
        except IOError:
            self.extract_from_zip(password=password, zip_filename=zip_filename,
                                  filename=filename)
            self.logger.info('Reading data from {}'.format(
                full_csv_filename))
            df = read_csv(full_csv_filename, encoding='utf-8')
        return df

    def extract_from_zip(self, password, zip_filename=ZIP_FILENAME,
                         filename_within_zip=None, filename=CSV_FILENAME):
        """Extract the datafile from the zip archive.

        The file will be read and written to a comma-separated file in the
        data directory.

        Parameters
        ----------
        password : str
            Password for the file in the zip archive.
        zip_filename : str, optional
            Local filename for the zip file
        filename_within_zip : str, optional
            Filename of the file within the zip-file. By default the first
            file will be extracted.
        filename : str, optional.
            Name of file to write.

        """
        df = self.read_from_zip(password=password, zip_filename=zip_filename,
                                filename=filename_within_zip)
        full_filename = self.full_filename(filename)
        self.logger.info('Writing data to {}'.format(full_filename))
        df.to_csv(full_filename, encoding='utf-8')

    def read_from_zip(self, password, zip_filename=ZIP_FILENAME,
                      filename=None):
        """Return data from zip file.

        Parameters
        ----------
        password : str
            Password for the file in the zip archive.
        zip_filename : str, optional
            Local filename for the zip file
        filename : str, optional
            Filename of the file within the zip-file. By default the first
            file will be extracted.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with information from the zip-file

        """
        # The file embedded in the Zip file has a varying number of elements in
        # each column making it difficult (impossible?) to read it directly
        # with pandas.

        columns = ['paragraph', 'sentence', 'word', 'lemma', 'tag', 'hmm']

        data = []
        full_filename = self.full_filename(zip_filename)
        self.logger.info('Reading data from zip file: {}'.format(
            full_filename))
        with ZipFile(full_filename) as zip_file:
            if filename is None:
                members = zip_file.namelist()
                filename = members[0]

            with zip_file.open(filename, pwd=password) as f:
                reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
                for idx, row in enumerate(reader, 1):
                    if len(row) == 0:
                        # Ignore empty line
                        pass
                    elif len(row) == 5 or len(row) == 6:
                        data.append([element.decode('utf-8')
                                     for element in row])
                    else:
                        self.logger.warn(
                            'Could not handle line {}: {}'.format(
                                idx, u' '.join(row)))

        df = DataFrame(data, columns=columns)
        return df

    def word_to_lemmas(self, word):
        """Convert word to list of lemmas.

        Parameters
        ----------
        word : str
            Word where the lemma should be found.

        Returns
        -------
        lemmas : list of collections.Counter
            List of Counters with counts for how often a lemma occurs for the
            word.

        Examples
        --------
        >>> ep = EParole()
        >>> counts = ep.word_to_lemmas(u'bager')
        >>> counts == {'bager': 2, 'bage': 1}
        True

        """
        if not self.word_to_lemmas_map:
            self.logger.info('Loading word to lemma map')
            df = self.read()
            for idx, row in df.iterrows():
                self.word_to_lemmas_map[row['word']].append(
                    row['lemma'].lower())
            self.word_to_lemmas_map = {
                key: Counter(value)
                for key, value in self.word_to_lemmas_map.items()}

        return self.word_to_lemmas_map.get(word.lower(), Counter())

    def word_to_lemma(self, word, return_if_not_exists='word'):
        """Convert word to a lemma.

        Parameters
        ----------
        word : str
            Word to be lemmatized. The word is converted to lowercase.
        return_if_not_exists : 'word', None, '' or raise, optional
            How to handle the case where the word does not appear in the
            dictionary.

        Examples
        --------
        >>> ep = EParole()
        >>> lemma = ep.word_to_lemma('bogen')
        >>> lemma == 'bog'
        True

        >>> ep.word_to_lemma('bager') == 'bager'
        True

        """
        lemmas = self.word_to_lemmas(word.lower())
        if len(lemmas) > 0:
            return lemmas.keys()[0]

        if return_if_not_exists == 'word':
            return word
        elif return_if_not_exists is None:
            return None
        elif return_if_not_exists == '':
            return ''
        elif return_if_not_exists == 'raise':
            raise KeyError('{} not in word to lemmas map'.format(word))


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

    if arguments['word-to-lemma']:
        word = arguments['<word>']
        if not isinstance(word, text_type):
            word = word.decode(input_encoding)

        eparole = EParole()
        lemma = eparole.word_to_lemma(word)
        write(output_file, lemma.encode(output_encoding) + b('\n'))


if __name__ == '__main__':
    main()
