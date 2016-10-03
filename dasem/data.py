"""Data."""

from os.path import join, split

from pandas import read_csv


def wordsim353(include_problems=False):
    """Read and return wordsim353 dataset.

    Parameters
    ----------
    include_problems : bool, optional
        Indicator for whether rows with problematic translations
        between Danish and English should be returned [default: False].

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with Danish wordsim353 data.

    Examples
    --------
    >>> df = wordsim353(include_problems=True)
    >>> df.shape
    (353, 6)

    """
    filename = join(split(__file__)[0], 'data', 'wordsim353-da',
                    'combined.csv')
    df = read_csv(filename)

    if include_problems:
        df = df[df.Problem != 1]

    return df
