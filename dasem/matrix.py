r"""Matrix.

Usage:
   dasem.matrix save-wikipedia-doc-term-matrix [options] <filename>
   dasem.matrix save-wikipedia-tdidf-doc-term-matrix [options] <filename>
   dasem.matrix save-wikipedia-doc-term-matrix-nmf-factorization \
       [options] <filename>

Options:
  --output-filename    Filename to write to
  -h --help            Help message
  --max-iter=<int>     Maximum number of iterations [default: 200]
  --max-n-pages=<int>  Maximum number of pages
  -v --verbose         Verbose debug messaging

Examples:
   python -m dasem.matrix save-wikipedia-doc-term-matrix \
       --max-n-pages=10 tmp.mtx

"""

from math import ceil, sqrt

from os.path import splitext

import json

from scipy import io

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer

from .wikipedia import XmlDumpFile


class Matrix(object):
    """Matrix."""

    def __init__(self, matrix, rows=None, columns=None):
        """Set up matrix values and row and column annotation.

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
        return "<Matrix({}x{})>".format(*self.matrix.shape)

    __repr__ = __str__

    @property
    def shape(self):
        """Return shape of matrix."""
        return self.matrix.shape

    def to_csr(self, copy=False):
        """Convert to CSR sparse matrix.

        Parameters
        ----------
        copy : bool
            Copy the matrix

        """
        if copy:
            # TODO copy lists and dicts
            matrix = Matrix(self.matrix.tocsr(copy=True),
                            rows=self.rows,
                            columns=self.columns)
            return matrix
        else:
            self.matrix = self.matrix.tocsr()
            return self

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

    def nmf(self, n_components=None, max_iter=200):
        """Compute the non-negative matrix factorization.

        Parameters
        ----------
        n_components : int or None
            Number of components.

        Returns
        -------
        W : Matrix
            Left factorization.
        H : Matrix
            Right factorization.

        """
        if n_components is None:
            n_components = int(ceil(sqrt(min(self.matrix.shape) / 2)))

        # Prepare the matrix for fast computations
        self.to_csr(copy=False)

        # Compute factorization
        factorizer = NMF(n_components=n_components, max_iter=max_iter)
        W_values = factorizer.fit_transform(self.matrix)

        # Prepare output
        components = ['Component {}'.format(n)
                      for n in range(1, W_values.shape[1]+1)]
        W = Matrix(W_values, rows=self.rows, columns=components)
        H = Matrix(factorizer.components_,
                   rows=components, columns=self.columns)
        return W, H

    def tfidf(self, copy=False):
        """Return tfidf-transformed matrix."""
        transformer = TfidfTransformer()
        if copy:
            matrix = Matrix(transformer.fit_transform(self.matrix),
                            rows=self.rows,
                            columns=self.columns)
            return matrix
        else:
            self.matrix = transformer.fit_transform(self.matrix)
            return self


def load_matrix(filename):
    """Load matrix from Matrix market file.

    Parameters
    ----------
    filename : str
        Input filename.

    """
    matrix_values = io.mmread(filename)
    with open(filename) as f:
        line = f.readline()  # Ignore first line
        line = f.readline()
    data = json.loads(line[1:])
    rows = data['rows']
    columns = data['columns']

    matrix = Matrix(matrix_values, rows, columns)
    return matrix


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)
    filename = arguments['<filename>']
    max_iter = int(arguments['--max-iter'])
    if arguments['--max-n-pages'] is None:
        max_n_pages = None
    else:
        max_n_pages = int(arguments['--max-n-pages'])
    verbose = verbose = arguments['--verbose']

    if arguments['save-wikipedia-doc-term-matrix']:
        dump_file = XmlDumpFile()
        matrix_values, rows, columns = dump_file.doc_term_matrix(
            max_n_pages=max_n_pages,
            verbose=verbose)
        matrix = Matrix(matrix_values, rows, columns)
        matrix.save(filename=filename)

    if arguments['save-wikipedia-tdidf-doc-term-matrix']:
        dump_file = XmlDumpFile()
        matrix_values, rows, columns = dump_file.doc_term_matrix(
            max_n_pages=max_n_pages,
            verbose=verbose)
        matrix = Matrix(matrix_values, rows, columns)
        matrix.tdidf()
        matrix.save(filename=filename)

    elif arguments['save-wikipedia-doc-term-matrix-nmf-factorization']:
        dump_file = XmlDumpFile()
        matrix_values, rows, columns = dump_file.doc_term_matrix(
            max_n_pages=max_n_pages,
            verbose=verbose)
        X = Matrix(matrix_values, rows, columns)
        if verbose:
            print('Matrix factorization on matrix sized {}x{}'.format(
                *X.shape))
        W, H = X.nmf(n_components=10, max_iter=max_iter)
        filename_base, filename_ext = splitext(filename)
        W.save(filename=filename_base + '-W' + filename_ext)
        H.save(filename=filename_base + '-H' + filename_ext)


if __name__ == '__main__':
    main()
