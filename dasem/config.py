"""config."""


from os.path import expanduser, join


def data_directory():
    """Return path for the data directory.

    Returns
    -------
    dirname : str
        Pathname for the data directory.

    """
    return join(expanduser('~'), 'dasem_data')
