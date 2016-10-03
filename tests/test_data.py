"""Test of data module."""

from dasem.data import wordsim353


def test_wordsim353():
    """Test of wordsim353 data."""
    df = wordsim353()
    assert not df.da1.isnull().any()
    assert not df.da2.isnull().any()

    mapper = {}
    for idx, row in df.iterrows():
        if row['Word 1'] in mapper:
            assert mapper[row['Word 1']] == row['da1']
        if row['Word 2'] in mapper:
            assert mapper[row['Word 2']] == row['da2']
        mapper[row['Word 1']] = row['da1']
        mapper[row['Word 2']] = row['da2']
