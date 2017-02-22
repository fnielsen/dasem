"""runeberg.

Usage:
  dasem.runeberg download-catalogue
  dasem.runeberg catalogue-as-csv

Description
-----------
Runeberg is a digital library with primarily Nordic texts. It is available from
http://runeberg.org/

"""


from __future__ import absolute_import, division, print_function

from os.path import join

from re import DOTALL, UNICODE, findall

import sys

from pandas import DataFrame

import requests

from .config import data_directory
from .utils import make_data_directory


CATALOGUE_URL = 'http://runeberg.org/katalog.html'

CATALOGUE_FILENAME = 'katalog.html'


def fix_author(author):
    """Change surname-firstname order.

    Parameters
    ----------
    author : str
        Author as string

    Returns
    -------
    fixed_author : str
        Changed author string.

    Examples
    --------
    >>> author = 'Lybeck, Mikael'
    >>> fix_author(author)
    'Mikael Lybeck'

    """
    author_parts = author.split(', ')
    if len(author_parts) == 2:
        fixed_author = author_parts[1] + ' ' + author_parts[0]
    else:
        fixed_author = author
    return fixed_author


class Runeberg(object):
    """Runeberg.

    Examples
    --------
    >>> runeberg = Runeberg()
    >>> catalogue = runeberg.catalogue()
    >>> danish_catalogue = catalogue.ix[catalogue.language == 'dk', :]
    >>> len(danish_catalogue) > 300
    True

    """

    def download_catalogue(self):
        """Download and store locally the Runeberg catalogue."""
        make_data_directory(data_directory(), 'runeberg')
        filename = join(data_directory(), 'runeberg', CATALOGUE_FILENAME)
        response = requests.get(CATALOGUE_URL)
        with open(filename, 'w') as f:
            f.write(response.content)

    def catalogue(self, fix_author=True):
        """Retrieve and parse Runeberg catalogue.

        Returns
        -------
        books : pandas.DataFrame
            Dataframe with book information.
        fix_author : bool, optional
            Determine if author names should be rearranged in firstname-surname
            order [default: True]

        """
        response = requests.get(CATALOGUE_URL)

        flags = DOTALL | UNICODE
        tables = findall(r'<table.*?</table>', response.text, flags=flags)
        rows = findall(r'<tr.*?</tr>', tables[1], flags=flags)

        books = []
        for row in rows[1:]:
            elements = findall('<td.*?</td>', row, flags=flags)
            book_id, title = findall(r'/(.*?)/">(.*?)<',
                                     elements[4], flags=flags)[0]
            try:
                author_id, author = findall(r'/authors/(.*?).html">(.*?)<',
                                            elements[6], flags=flags)[0]
            except:
                author_id, author = '', ''
            if fix_author:
                # fix_author name collision. TODO
                author = globals()['fix_author'](author)
            book = {
                'type': findall(r'alt="(.*?)">', elements[0], flags=flags)[0],
                'book_id': book_id,
                'title': title,
                'author_id': author_id,
                'author': author,
                'year': elements[8][15:-5],
                'language': elements[10][-9:-7]
            }
            books.append(book)
        return DataFrame(books)


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)
    if sys.stdout.encoding is None:
        encoding = 'utf-8'
    else:
        encoding = sys.stdout.encoding

    runeberg = Runeberg()

    if arguments['download-catalogue']:
        runeberg.download_catalogue()

    elif arguments['catalogue-as-csv']:
        print(runeberg.catalogue().to_csv(encoding=encoding))


if __name__ == '__main__':
    main()
