"""config"""


from os.path import expanduser, join


def data_directory():
    return join(expanduser('~'), 'dasem_data')
