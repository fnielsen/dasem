"""utils.

Usage:
  dasem.utils make-data-directory

"""

from __future__ import absolute_import, division, print_function

import errno

from os import makedirs
from os.path import join

from .config import data_directory


def make_data_directory(*args):
    """Make data directory.

    The data_directory is by default `dasem_data`. If `directory` is None then
    this directory is created if it does not already exist.

    Parameters
    ----------
    directory : str or None
        Name of directory

    """
    if len(args) == 0:
        make_data_directory(data_directory())
    else:
        try:
            makedirs(join(*args))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    if arguments['make-data-directory']:
        make_data_directory()


if __name__ == '__main__':
    main()
