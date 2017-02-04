"""Data.

Functions to read datasets from the data subdirectory.

"""

from os.path import join, split

from pandas import read_csv


def four_words():
    """Read and return four words odd-one-out dataset.

    Returns
    -------
    >>> df = four_words()
    >>> df.ix[0, 'word4'] == 'stol'
    True

    """
    filename = join(split(__file__)[0], 'data', 'four_words.csv')
    df = read_csv(filename, encoding='utf-8')
    return df


def verbal_analogies():
    """Read and return verbal analogies dataset.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with verbal analogies.

    Examples
    --------
    >>> df = verbal_analogies()
    >>> df.ix[0, :].tolist() == ['mand', 'kvinde', 'dreng', 'pige']
    True

    """
    filename = join(split(__file__)[0], 'data', 'verbal_analogies.csv')
    df = read_csv(filename, encoding='utf-8')
    return df


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

    References
    ----------
    The WordSimilarity-353 Test Collection,
    http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/

    """
    filename = join(split(__file__)[0], 'data', 'wordsim353-da',
                    'combined.csv')
    df = read_csv(filename, encoding='utf-8')

    if not include_problems:
        df = df[df.Problem != 1]

    return df
