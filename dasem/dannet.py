"""dasem.dannet - Interface to DanNet.

Usage:
  dasem.dannet build-sqlite-database [options]
  dasem.dannet doc2vec-most-similar [options] <document>
  dasem.dannet download [options]
  dasem.dannet fasttext-vector [options] <word>
  dasem.dannet get-all-sentences [options]
  dasem.dannet get-all-tokenized-sentences [options]
  dasem.dannet show-glossary <word> [options]
  dasem.dannet fasttext-most-similar [options] <word>
  dasem.dannet show [options] <dataset>
  dasem.dannet train-and-save-doc2vec [options]
  dasem.dannet train-and-save-fasttext [options]

Options:
  --debug             Debug messages
  -h --help           Help message
  -i --input=<file>   Input filename
  --ie=encoding       Input encoding [default: utf-8]
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  -v --verbose  Verbose informational messages

Description:
  This module handles DanNet, the Danish wordnet.

  The `get-all-sentences` command will get all usage example sentences
  from the synsets.

  The script will automagically download the data from the DanNet homepage.

  words.csv:
     3-columns: (id, form, pos), e.g., (50001462, druemost, Noun)
     The id is found in the wordsenses.csv. It is for the lexical entry

  wordsenses.csv:
     4-columns (wordsense_id, word_id, synset_id, ?), e.g.,
     (22005172, 50001462, 66967, )

  For instance, relations.csv describes 2355 (gruppe_1; samling_3) as being a
  hyponym of 20633 (DN:TOP) and synonym of WordNet's ENG20-08119921-n.

References:
  http://wordnet.dk/

"""


from __future__ import absolute_import, division, print_function

import csv

import json

import logging

import os
from os import write
from os.path import isfile, join, sep, splitext

import re

import sqlite3

from shutil import copyfileobj

import signal

from sys import version_info

from zipfile import ZipFile

from gensim.models.doc2vec import TaggedDocument

from db import DB

from nltk.stem.snowball import DanishStemmer
from nltk.tokenize import WordPunctTokenizer

from pandas import read_csv, DataFrame
from pandas.io.common import CParserError

import requests

from six import b, text_type, u

from . import models
from .config import data_directory
from .corpus import Corpus
from .utils import make_data_directory


BASE_URL = 'http://www.wordnet.dk/'

DANNET_FILENAME = 'DanNet-2.2_csv.zip'

DANNET_SQLITE_FILENAME = splitext(DANNET_FILENAME)[0] + '.db'

DANNET_CSV_ZIP_URL = 'http://www.wordnet.dk/DanNet-2.2_csv.zip'


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
        directory = join(data_directory(), 'dannet')
        return directory


class Dannet(Corpus, DataDirectoryMixin):
    """Dannet.

    Using the module will automagically download the data from the Dannet
    homepage (http://www.wordnet.dk).

    Attributes
    ----------
    db : db.DB
        Database access through the db.py interface.

    Examples
    --------
    >>> dannet = Dannet()
    >>> dannet.db.tables.words
    +---------------------------------------------------+
    |                       words                       |
    +---------+---------+--------------+----------------+
    | Column  | Type    | Foreign Keys | Reference Keys |
    +---------+---------+--------------+----------------+
    | index   | INTEGER |              |                |
    | word_id | TEXT    |              |                |
    | form    | TEXT    |              |                |
    | pos     | TEXT    |              |                |
    +---------+---------+--------------+----------------+

    >>> # From README
    >>> query = '''
    ... SELECT w.form, ws.register, s.synset_id, s.gloss, s.ontological_type
    ... FROM synsets s, wordsenses ws, words w
    ... WHERE s.synset_id = ws.synset_id
    ...   AND ws.word_id = w.word_id
    ...   AND w.form = 'spand';'''
    >>> 'bil' in dannet.db.query(query).gloss[0]
    True

    >>> # Danish nouns
    >>> dannet = Dannet()
    >>> query = "select w.form from words w where w.pos = 'Noun'"
    >>> nouns = set(dannet.db.query(query).form)
    >>> 'guitar' in nouns
    True
    >>> 'guitaren' in nouns
    False
    >>> len(nouns)
    48404

    References
    ----------
    - http://www.wordnet.dk

    """

    def __init__(self):
        """Initialize logger and and database."""
        self.logger = logging.getLogger(__name__ + '.Dannet')
        self.logger.addHandler(logging.NullHandler())

        self.logger.debug('Initializing tokenizer and stemmer')
        self.word_tokenizer = WordPunctTokenizer()
        self.stemmer = DanishStemmer()

        self._db = None

    @property
    def db(self):
        """Return a db.py instance with DanNet data."""
        if self._db is not None:
            return self._db

        full_filename = self.full_filename(DANNET_SQLITE_FILENAME)
        self.logger.info('Trying to read database file {}'.format(
            full_filename))
        try:
            self._db = DB(filename=full_filename, dbtype='sqlite')
            if not hasattr(self._db.tables, 'words'):
                self.logger.debug('Database is empty')
                # There is no content in the database
                raise Exception('Not initialized')
        except:
            self.build_sqlite_database()
            self._db = DB(filename=full_filename, dbtype='sqlite')
        return self._db

    def download(self, filename=DANNET_FILENAME, redownload=False):
        """Download data."""
        local_filename = join(self.data_directory(), filename)
        if not redownload and isfile(local_filename):
            message = 'Not downloading as corpus already download to {}'
            self.logger.debug(message.format(local_filename))
            return

        self.make_data_directory()
        url = BASE_URL + filename
        self.logger.info('Downloading from URL {} to {}'.format(
            url, local_filename))
        response = requests.get(url, stream=True)
        with open(local_filename, 'wb') as fid:
            copyfileobj(response.raw, fid)
        self.logger.debug('Corpus downloaded'.format())

    def full_filename(self, filename=DANNET_FILENAME):
        """Prepend data directory path to filename.

        Parameters
        ----------
        filename : str
            Filename of local Dannet file.

        Returns
        -------
        full_filename : str
            Filename with full directory path information.

        """
        if sep in filename:
            return filename
        else:
            return join(data_directory(), 'dannet', filename)

    def glossary(self, word):
        """Return glossary for word.

        Parameters
        ----------
        word : str
            Query word.

        Returns
        -------
        glossary : list of str
            List of distinct strings from `gloss` field of synsets which
            form matches the query word.

        Examples
        --------
        >>> dannet = Dannet()
        >>> len(dannet.glossary('virksomhed')) == 3
        True

        """
        query_template = u("""
            SELECT DISTINCT s.gloss
            FROM synsets s, wordsenses ws, words w
            WHERE s.synset_id = ws.synset_id AND
                ws.word_id = w.word_id AND w.form = '{word}';""")
        query = query_template.format(
            word=word.replace('\\', '\\\\').replace("'", "\\'"))
        self.logger.debug(u('Querying with {}').format(
            query.replace('\n', ' ')))
        glossary = list(self.db.query(query).gloss)
        return glossary

    def iter_sentences(self):
        """Iterate over sentences in the synsets examples.

        The synsets definitions have examples of word usages. There might be
        several examples for some synsets. This function iterates over all the
        sentences.

        Yields
        ------
        sentence : str
            Sentence.

        """
        use_pattern = re.compile(r'\(Brug: (".+?")\)', flags=re.UNICODE)
        quote_pattern = re.compile(r'"(.+?)"(?:; "(.+?)")*', flags=re.UNICODE)
        synsets = self.read_synsets()
        self.logger.debug('Iterating over sentences')
        for gloss in synsets.gloss:
            use_matches = use_pattern.findall(gloss)
            if use_matches:
                quote_matches = quote_pattern.findall(use_matches[0])
                for parts in quote_matches[0]:
                    sentences = parts.split(' || ')
                    for sentence in sentences:
                        if sentence:
                            yield sentence.replace('[', '').replace(']', '')

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

    def read_zipped_csv_file(self, filename, zip_filename=DANNET_FILENAME):
        """Read a zipped csv DanNet file.

        The csv file is read with the 'latin_1' encoding.

        Parameters
        ----------
        filename : str
            Filename of the file within the zip file.
        zip_filename : str
            Filename of the zip file. This is expanded as it expect the data
            to be in the data directory.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with the data from the csv file.

        """
        full_zip_filename = self.full_filename(zip_filename)

        if not isfile(full_zip_filename):
            self.logger.info('File {} not downloaded'.format(zip_filename))
            self.download()

        full_filename = join(splitext(zip_filename)[0], filename)

        self.logger.info('Reading from {}'.format(full_zip_filename))
        zip_file = ZipFile(full_zip_filename)
        try:
            df = read_csv(zip_file.open(full_filename),
                          sep='@', encoding='latin_1', header=None)
        except CParserError:
            self.logger.debug('Reading of csv with Pandas failed')
            # Bad csv file with unquoted "@" in line 19458 and 45686
            # in synsets.csv
            with zip_file.open(full_filename) as fid:
                # Major problem with getting Python2/3 compatibility
                if version_info[0] == 2:
                    csv_file = csv.reader(fid, delimiter='@')
                    rows = []
                    for row in csv_file:
                        if len(row) == 6:
                            row = [row[0], row[1], row[2] + '@' + row[3],
                                   row[4], row[5]]
                        row = [elem.decode('latin_1') for elem in row]
                        rows.append(row)
                else:
                    # Encoding problem handle with
                    # https://stackoverflow.com/questions/36971345
                    lines = (line.decode('latin_1') for line in fid)
                    csv_file = csv.reader(lines, delimiter='@')
                    rows = []
                    for row in csv_file:
                        if len(row) == 6:
                            row = [row[0], row[1], row[2] + '@' + row[3],
                                   row[4], row[5]]
                        rows.append(row)
            df = DataFrame(rows)

        # Drop last column which always seems to be superfluous
        df = df.iloc[:, :-1]
        self.logger.debug('Read {}x{} data from csv'.format(*df.shape))

        return df

    def make_data_directory(self):
        """Make data directory for LCC."""
        make_data_directory(self.data_directory())

    def read_relations(self, zip_filename=DANNET_FILENAME):
        """Read relations CSV file.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with columns synset_id, name, name2, value, taxonomic,
            inheritance_comment.

        """
        df = self.read_zipped_csv_file(
            'relations.csv', zip_filename=zip_filename)
        df.columns = ['synset_id', 'name', 'name2', 'value', 'taxonomic',
                      'inheritance_comment']
        return df

    def read_synset_attributes(self, zip_filename=DANNET_FILENAME):
        """Read synset attributes CSV file.

        Parameters
        ----------
        zip_filename : str
            Filename for the zip file with the CSV file.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with columns synset_id, type and value.

        """
        df = self.read_zipped_csv_file(
            'synset_attributes.csv', zip_filename=zip_filename)
        df.columns = ['synset_id', 'type', 'value']
        return df

    def read_synsets(self, zip_filename=DANNET_FILENAME):
        """Read synsets CSV file.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with columns id, label, gloss, ontological_type.

        Examples
        --------
        >>> dannet = Dannet()
        >>> df = dannet.read_synsets()
        >>> 'label' in df.columns
        True

        """
        df = self.read_zipped_csv_file(
            'synsets.csv', zip_filename=zip_filename)
        # import pdb
        # pdb.set_trace()
        df.columns = ['synset_id', 'label', 'gloss', 'ontological_type']
        return df

    def read_words(self, zip_filename=DANNET_FILENAME):
        """Read words from CSV file.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with id, form and pos columns.

        """
        df = self.read_zipped_csv_file('words.csv', zip_filename=zip_filename)
        df.columns = ['word_id', 'form', 'pos']
        return df

    def read_wordsenses(self, zip_filename=DANNET_FILENAME):
        """Read wordsenses data file.

        Returns
        -------
        df : pandas.DataFrame
           Dataframe with the columns wordsense_id, word_id, synset_id and
           register.

        """
        df = self.read_zipped_csv_file(
            'wordsenses.csv', zip_filename=zip_filename)
        df.columns = ['wordsense_id', 'word_id', 'synset_id', 'register']
        return df

    def build_sqlite_database(
            self, filename=DANNET_SQLITE_FILENAME,
            zip_filename=DANNET_FILENAME, if_exists='replace'):
        """Build SQLite database with DanNet data.

        This function will read the comma-separated values files and add the
        information to a SQLite database stored in the data directory under
        dannet.

        Execution of this function will typically take a couple of seconds.

        Parameters
        ----------
        filename : str, optional
            Filename of the SQLite file.
        zip_filename : str, optional
            Filename of CSV file.
        if_exists : bool, optional
            Determines whether the database tables should be overwritten
            (replace) [default: replace]

        """
        tables = [
            ('relations', self.read_relations),
            ('synset_attributes', self.read_synset_attributes),
            ('synsets', self.read_synsets),
            ('words', self.read_words),
            ('wordsenses', self.read_wordsenses)
        ]

        full_filename = self.full_filename(filename)
        self.logger.info('Building "{full_filename}" sqlite file'.format(
            full_filename=full_filename))

        with sqlite3.connect(full_filename) as connection:
            for table, method in tables:
                df = method(zip_filename=zip_filename)
                self.logger.info('Writing "{table}" table'.format(table=table))
                df.to_sql(table, con=connection, if_exists=if_exists)


class TaggedDocumentsIterable(object):
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

    def __iter__(self):
        """Restart and return iterable."""
        dannet = Dannet()
        for n, sentence_words in enumerate(dannet.iter_sentence_words(
                lower=self.lower, stem=self.stem)):
            tagged_document = TaggedDocument(sentence_words, [n])
            yield tagged_document


class Doc2Vec(DataDirectoryMixin, models.Doc2Vec):
    """Doc2Vec model for the Dannet corpus."""

    def iterable_tagged_documents(self, lower=True, stem=False):
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
        tagged_documents = TaggedDocumentsIterable(lower=lower, stem=stem)
        return tagged_documents


class FastText(DataDirectoryMixin, models.FastText):
    """FastText on Dannet corpus.

    It requires that a file called `sentences.txt` is available in the data
    directory.

    """

    pass


def main():
    """Handle command-line input."""
    from docopt import docopt

    arguments = docopt(__doc__)

    # Ignore broken pipe errors
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    logging_level = logging.WARN
    if arguments['--verbose']:
        logging_level = logging.INFO
    if arguments['--debug']:
        logging_level = logging.DEBUG

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
        logger.debug('Writing to file {}'.format(output_filename))
    else:
        # stdout
        output_file = 1

    input_filename = arguments['--input']

    output_encoding = arguments['--oe']
    input_encoding = arguments['--ie']

    if arguments['show']:
        dannet = Dannet()

        if arguments['<dataset>'] == 'relations':
            dataset = dannet.read_relations()
        elif arguments['<dataset>'] == 'synsets':
            dataset = dannet.read_synsets()
        elif arguments['<dataset>'] == 'synset_attributes':
            dataset = dannet.read_synset_attributes()
        elif arguments['<dataset>'] == 'words':
            dataset = dannet.read_words()
        elif arguments['<dataset>'] == 'wordsenses':
            dataset = dannet.read_wordsenses()
        else:
            raise ValueError('Wrong <dataset>')

        # https://stackoverflow.com/questions/42628069
        if version_info[0] == 2:
            write(output_file,
                  dataset.to_csv(encoding=output_encoding, index=False))
        else:
            output = dataset.to_csv(index=False)
            write(output_file, output.encode(output_encoding))

    elif arguments['build-sqlite-database']:
        dannet = Dannet()
        dannet.build_sqlite_database()

    elif arguments['doc2vec-most-similar']:
        document = arguments['<document>']
        if not isinstance(document, text_type):
            document = document.decode(input_encoding)

        doc2vec = Doc2Vec()
        for word, similarity in doc2vec.most_similar(document.split()):
            write(output_file, word.encode('utf-8') + b('\n'))

    elif arguments['download']:
        dannet = Dannet()
        dannet.download()

    elif arguments['fasttext-vector']:
        word = arguments['<word>']
        if not isinstance(word, text_type):
            word = word.decode(input_encoding)

        fast_text = FastText()
        print(json.dumps(fast_text.word_vector(word).tolist()))

    elif arguments['get-all-sentences']:
        dannet = Dannet()
        for sentence in dannet.iter_sentences():
            write(output_file, sentence.encode(output_encoding) + b('\n'))

    elif arguments['get-all-tokenized-sentences']:
        dannet = Dannet()
        for sentence in dannet.iter_tokenized_sentences():
            write(output_file, sentence.encode(output_encoding) + b('\n'))

    elif arguments['fasttext-most-similar']:
        word = arguments['<word>']
        if not isinstance(word, text_type):
            word = word.decode(input_encoding)

        fast_text = FastText()
        for word, similarity in fast_text.most_similar(word):
            write(output_file, word.encode('utf-8') + b('\n'))

    elif arguments['show-glossary']:
        word = arguments['<word>']
        if not isinstance(word, text_type):
            word = word.decode(input_encoding)

        dannet = Dannet()
        glossary = dannet.glossary(word)
        for gloss in glossary:
            write(output_file, gloss.encode('utf-8') + b('\n'))

    elif arguments['train-and-save-doc2vec']:
        doc2vec = Doc2Vec()
        if input_filename:
            doc2vec.train(input_filename=input_filename)
        else:
            doc2vec.train()

    elif arguments['train-and-save-fasttext']:
        fast_text = FastText()
        if input_filename:
            fast_text.train(input_filename=input_filename)
        else:
            fast_text.train()

    else:
        # Coding error if we arrive here
        assert False


if __name__ == '__main__':
    main()
