r"""gutenberg.

Usage:
  dasem.gutenberg get <id>
  dasem.gutenberg get-all-texts [options]
  dasem.gutenberg list-all-ids
  dasem.gutenberg list

Options:
  --oe=encoding  Output encoding [default: utf-8']

Description:
  This is an interface to Danish texts on Gutenberg.

  There is restriction on how the data should be downloaded from Gutenberg.
  This is stated on their homepage. Download of all the Danish language text
  must be done in the below way.

  wget -w 2 -m -H \
    "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=da"

  Danish works in Project Gutenberg are to some extent indexed on Wikidata. The
  works can be queried with:

    select ?work ?workLabel where {
      ?work wdt:P2034 ?gutenberg .
      ?work wdt:P364 wd:Q9035 .
      service wikibase:label { bd:serviceParam wikibase:language "da" }
    }

  The `list` command will query Wikidata.

References:
  https://www.gutenberg.org/wiki/Gutenberg:Information_About_Robot_Access_to_our_Pages

"""


from __future__ import print_function

import re

from os import walk
from os.path import join

from zipfile import ZipFile

from pandas import DataFrame

import requests

from sparql import Service

from .config import data_directory


SPARQL_QUERY = """
SELECT ?work ?workLabel ?authorLabel ?gutenberg WHERE {
  ?work wdt:P2034 ?gutenberg.
  ?work wdt:P407 wd:Q9035 .
  OPTIONAL { ?work wdt:P50 ?author . }
  service wikibase:label { bd:serviceParam wikibase:language "da" }
}
"""


def extract_text(text):
    """Extract text from downloaded text.

    Parameters
    ----------
    text : str
        Complete text from Gutenberg file.

    Returns
    -------
    extracted_text : str
        Extracted body.

    Description
    -----------
    This function attempts to extract the body of the the returned text.

    Start:

      *** START OF THIS PROJECT GUTENBERG EBOOK ... ***

      Some multiple lines of text

    The button of the returned text is the GPL license. The postamble is
    indictated there is three stars and perhaps a whitespace followed by
    "END OF THIS ..." and sometimes "END OF THE ...", e.g.,:

    "*** END OF THIS PROJECT GUTENBERG EBOOK ... ***"

    before this indication there might be further metadata, e.g.:

    "End of the Project Gutenberg EBook of ..."

    This postamble seems not always to be present. It might be split over the
    two last sentences.

    """
    # TODO: There is still some text to be dealt with.
    matches = re.findall(
        (r"^\*\*\* ?START OF TH.+?$"
         r"(.+?)"  # The text to capture
         r"^\*\*\* ?END OF TH.+?$"),
        text, flags=re.DOTALL | re.MULTILINE | re.UNICODE)
    body = matches[0].strip()
    return body


def get_list_from_wikidata():
    """Get list of works from Wikidata.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with information from Wikidata.

    """
    service = Service("https://query.wikidata.org/sparql", method="GET")
    result = service.query(SPARQL_QUERY)
    df = DataFrame(result.fetchall(), columns=result.variables)
    return df


def get_text_by_id(id):
    """Get text from Gutenberg based on id.

    Project Gutenberg sets a restriction on the way that text on their site
    must be downloaded. This function does not honor the restriction, so the
    function should be used with care.

    Parameters
    ----------
    id : int or str
        Identifier.

    Returns
    -------
    text : str or None
        The text. Returns none if the page does not exist.

    References
    ----------
    https://www.gutenberg.org/wiki/Gutenberg:Information_About_Robot_Access_to_our_Pages

    """
    url = "http://www.gutenberg.org/ebooks/{id}.txt.utf-8".format(id=id)
    response = requests.get(url)
    return response.content


class Gutenberg(object):
    """Gutenberg.

    Encoding
    --------
    10218 is encoded in "ISO Latin-1". This is stated with the line
    "Character set encoding: ISO Latin-1" in the header of the data file.

    """

    def __init__(self):
        """Setup data directory."""
        self.data_directory = join(data_directory(), 'gutenberg',
                                   'www.gutenberg.lib.md.us')

    def get_all_ids(self):
        """Get all Gutenberg text ids from mirrored data.

        Returns
        -------
        ids : list of str
            List of Gutenberg ebook identifiers.

        Examples
        --------
        >>> gutenberg = Gutenberg()
        >>> '38080' in gutenberg.get_all_ids()
        True

        """
        ids = []
        for root, dirs, files in walk(self.data_directory):
            for file in files:
                if file.endswith('-8.zip'):
                    ids.append(file[:-6])
        return ids

    def get_text_by_id(self, id, extract_body=True):
        """Get text from mirrored Gutenberg archive.

        This function requires that the texts have been mirrored.

        Parameters
        ----------
        id : str or integer
            Gutenberg ebook identifier.

        Returns
        -------
        text : str
            Extracted text.

        """
        # Example on subdirectory structure:
        # www.gutenberg.lib.md.us/4/4/9/6/44967
        s = str(id)
        l = list(s)
        if len(l) > 4:
            directory = join(self.data_directory, l[0], l[1], l[2], l[3], s)
        else:
            # For instance, id=9264 has only four-level subdirectories.
            # This might be because it is only 4 characters long
            directory = join(self.data_directory, l[0], l[1], l[2], s)

        zip_filename = join(directory, s + '-8.zip')
        with ZipFile(zip_filename) as zip_file:
            filename = join(s, s + '-8.txt')
            try:
                with zip_file.open(filename) as f:
                    encoded_text = f.read()
            except KeyError:
                # There might be zip files where the data file is in the root
                filename = s + '-8.txt'
                with zip_file.open(filename) as f:
                    encoded_text = f.read()

        if re.search(r'^Character set encoding: ISO-8859-1',
                     encoded_text, flags=re.DOTALL | re.MULTILINE):
            text = encoded_text.decode('ISO-8859-1')
        elif re.search(
                r'^Character set encoding: ISO Latin-1',
                encoded_text, flags=re.DOTALL | re.MULTILINE):
            text = encoded_text.decode('Latin-1')
        else:
            raise LookupError('Unknown encoding for file {}'.format(filename))

        if extract_body:
            extracted_text = extract_text(text)
            return extracted_text
        else:
            return text

    def iter_texts(self):
        """Yield texts.

        Yields
        ------
        text : str
            Text.

        """
        for id in self.get_all_ids():
            yield self.get_text_by_id(id)


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)
    encoding = arguments['--oe']

    if arguments['get']:
        gutenberg = Gutenberg()
        text = gutenberg.get_text_by_id(arguments['<id>'])
        print(text)

    elif arguments['get-all-texts']:
        gutenberg = Gutenberg()
        for text in gutenberg.iter_texts():
            print(text.encode(encoding))

    elif arguments['list-all-ids']:
        gutenberg = Gutenberg()
        ids = gutenberg.get_all_ids()
        for id in ids:
            print(id)

    elif arguments['list']:
        df = get_list_from_wikidata()
        print(df.to_csv(encoding='utf-8'))


if __name__ == '__main__':
    main()
