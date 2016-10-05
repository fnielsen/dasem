"""runeberg.

Usage:
  dasem.runeberg


"""


from __future__ import division, print_function

from re import DOTALL, UNICODE, findall

import sys

from pandas import DataFrame

import requests


CATALOGUE_URL = 'http://runeberg.org/katalog.html'


class Runeberg(object):

    def catalogue(self):
        """Retrieve and parse Runeberg catalogue.

        Returns
        -------
        books : pandas.DataFrame
            Dataframe with book information.

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
    runeberg = Runeberg()
    print(runeberg.catalogue().to_csv(encoding=sys.stdout.encoding))


if __name__ == '__main__':
    main()
