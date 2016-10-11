"""dannet.

Usage:
  dasem.dannet show <dataset>


Description
-----------
words.csv:
   3-columns: (id, form, pos), e.g., (50001462, druemost, Noun)
   The id is found in the wordsenses.csv. It is for the lexical entry

wordsenses.csv:
   4-columns (wordsense_id, word_id, synset_id, ?), e.g.,
   (22005172, 50001462, 66967, )


For instance, relations.csv describes 2355 (gruppe_1; samling_3) as being a
hyponym of 20633 (DN:TOP) and synonym of WordNet's ENG20-08119921-n.


"""


from __future__ import absolute_import, division, print_function

import csv

from os.path import join, sep, splitext

import sys

from zipfile import ZipFile

from pandas import read_csv, DataFrame
from pandas.io.common import CParserError

from .config import data_directory

DANNET_FILENAME = 'DanNet-2.2_csv.zip'


class Dannet(object):

    def full_zip_filename(self, filename=DANNET_FILENAME):
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
        zip_file = ZipFile(self.full_zip_filename(zip_filename))
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
        df = self.read_zipped_csv_file(
            'synset_attributes.csv', zip_filename=zip_filename)
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
        df.columns = ['id', 'label', 'gloss', 'ontological_type']
        return df

    def read_words(self, zip_filename=DANNET_FILENAME):
        """Read words from CSV file.

        Returns
        -------
        df : pandas.DataFrame
            Dataframe with id, form and pos columns.

        """
        df = self.read_zipped_csv_file('words.csv', zip_filename=zip_filename)
        df.columns = ['id', 'form', 'pos']
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


def main():
    """Handle command-line input."""
    from docopt import docopt

    arguments = docopt(__doc__)

    encoding = sys.stdout.encoding
    if not encoding:
        # In Python2 sys.stdout.encoding is set to None for piped output
        encoding = 'utf-8'

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
    print(dataset.to_csv(encoding=encoding, index=False))


if __name__ == '__main__':
    main()
