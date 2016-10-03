"""Data."""

from os.path import join, split

from pandas import read_csv


def wordsim353():
    """Read and return wordsim353 dataset.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with Danish wordsim353 data.

    Examples
    --------
    >>> df = wordsim353()
    >>> df.shape
    (353, 6)

    """
    filename = join(split(__file__)[0], 'data', 'wordsim353-da',
                    'combined.csv')
    df = read_csv(filename)
    return df
