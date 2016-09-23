"""Matrix.

Usage:
   dasem.docterm save-wikipedia-doc-term-matrix [options] <filename>

Options:
  --output-filename    Filename to write to
  -h --help            Help message
  --max-n-pages=<int>  Maximum number of pages
  -v --verbose         Verbose debug messaging


"""

import json

from scipy import io

from .wikipedia import XmlDumpFile


class DocTermMatrix(object):

    def __init__(self, matrix, rows, columns):
        """Setup matrix values and row and column annotation.

        Parameters
        ----------
        matrix : matrix from scipy or numpy
            Values
        rows : list of str
            Row annotation.
        columns : list of str
            Column annotation.

        """
        self.matrix = matrix
        self.rows = rows
        self.columns = columns

    def __str__(self):
        """Return string representation."""
        return "<DocTermMatrix({}x{})>".format(*self.matrix.shape)

    __repr__ = __str__

    def save(self, filename):
        """Save matrix to a Matrix Market file.

        Parameters
        ----------
        filename : str
            Output filename.

        """
        comment = json.dumps({'rows': self.rows,
                              'columns': self.columns})
        io.mmwrite(filename, self.matrix, comment=comment)


def load_doc_term_matrix(filename):
    """Load matrix from Matrix market file.

    Parameters
    ----------
    filename : str
        Input filename.

    """
    matrix = io.mmread(filename)
    with open(filename) as f:
        line = f.readline()  # Ignore first line
        line = f.readline()
    data = json.loads(line[1:])
    rows = data['rows']
    columns = data['columns']

    doc_term_matrix = DocTermMatrix(matrix, rows, columns)
    return doc_term_matrix


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    if arguments['save-wikipedia-doc-term-matrix']:
        dump_file = XmlDumpFile()
        matrix, rows, columns = dump_file.doc_term_matrix(
            max_n_pages=int(arguments['--max-n-pages']),
            verbose=arguments['--verbose'])
        doc_term_matrix = DocTermMatrix(matrix, rows, columns)
        doc_term_matrix.save(
            filename=arguments['<filename>'])


if __name__ == '__main__':
    main()
