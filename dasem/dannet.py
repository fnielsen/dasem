"""dasem.dannet.

Usage:
  dasem.dannet show <dataset>
  dasem.dannet build-sqlite-database [options]

Options:
  --debug       Debug messages
  -h --help     Help message
  -v --verbose  Verbose informational messages


Description
-----------

This module handles DanNet, the Danish wordnet.


words.csv:
   3-columns: (id, form, pos), e.g., (50001462, druemost, Noun)
   The id is found in the wordsenses.csv. It is for the lexical entry

wordsenses.csv:
   4-columns (wordsense_id, word_id, synset_id, ?), e.g.,
   (22005172, 50001462, 66967, )


For instance, relations.csv describes 2355 (gruppe_1; samling_3) as being a
hyponym of 20633 (DN:TOP) and synonym of WordNet's ENG20-08119921-n.


Examples
--------
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
http://wordnet.dk/

"""


from __future__ import absolute_import, division, print_function

import csv

import logging

from os.path import join, sep, splitext

import sys

import sqlite3

from zipfile import ZipFile

from db import DB

from pandas import read_csv, DataFrame
from pandas.io.common import CParserError

from .config import data_directory


DANNET_FILENAME = 'DanNet-2.2_csv.zip'

DANNET_SQLITE_FILENAME = splitext(DANNET_FILENAME)[0] + '.db'


class Dannet(object):
    """Dannet.

    Attributes
    ----------
    db : db.DB
        Database access through the db.py interface.

    Examples
    --------
    >>> dannet = Dannet()
    >>> dannet.db.tables.words
    +--------------------------------------------------+
    |                      words                       |
    +--------+---------+--------------+----------------+
    | Column | Type    | Foreign Keys | Reference Keys |
    +--------+---------+--------------+----------------+
    | index  | INTEGER |              |                |
    | id     | TEXT    |              |                |
    | form   | TEXT    |              |                |
    | pos    | TEXT    |              |                |
    +--------+---------+--------------+----------------+

    >>> query = '''  # From README
    ... SELECT w.form, ws.register, s.synset_id, s.gloss, s.ontological_type
    ... FROM synsets s, wordsenses ws, words w
    ... WHERE s.synset_id = ws.synset_id
    ...   AND ws.word_id = w.word_id
    ...   AND w.form = 'spand';'''
    >>> 'bil' in dannet.db.query(query).gloss[0]
    True

    """

    def __init__(self, logging_level=logging.WARN):
        """Initialize logger and and database."""
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.NullHandler())
        self.logger.setLevel(logging_level)

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
        except:
            self.build_sqlite_database()
            self._db = DB(filename=full_filename, dbtype='sqlite')
        return self._db

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
        full_filename = join(splitext(zip_filename)[0], filename)
        zip_file = ZipFile(self.full_filename(zip_filename))
        try:
            df = read_csv(zip_file.open(full_filename),
                          sep='@', encoding='latin_1', header=None)
        except CParserError:
            # Bad csv file with unquoted "@" in line 19458 and 45686
            # in synsets.csv
            with zip_file.open(full_filename) as f:
                csv_file = csv.reader(f, delimiter='@')
                rows = []
                for row in csv_file:
                    if len(row) == 6:
                        row = [row[0], row[1], row[2] + '@' + row[3],
                               row[4], row[5]]
                    row = [element.decode('latin_1') for element in row]
                    rows.append(row)
            df = DataFrame(rows)

        # Drop last column which always seems to be superfluous
        df = df.iloc[:, :-1]

        return df

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

        """
        df = self.read_zipped_csv_file(
            'synsets.csv', zip_filename=zip_filename)
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


def main():
    """Handle command-line input."""
    from docopt import docopt

    logging.basicConfig()

    arguments = docopt(__doc__)

    encoding = sys.stdout.encoding
    if not encoding:
        # In Python2 sys.stdout.encoding is set to None for piped output
        encoding = 'utf-8'

    logging_level = logging.WARN
    if arguments['--verbose']:
        logging_level = logging.INFO
    if arguments['--debug']:
        logging_level = logging.DEBUG

    dannet = Dannet(logging_level=logging_level)

    if arguments['show']:
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
        print(dataset.to_csv(encoding=encoding, index=False))

    elif arguments['build-sqlite-database']:
        dannet.build_sqlite_database()


if __name__ == '__main__':
    main()
