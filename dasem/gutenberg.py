"""gutenberg.

Usage:
  dasem.gutenberg get <id>
  dasem.gutenberg list

"""


from __future__ import print_function

import re

from pandas import DataFrame

import requests

from sparql import Service


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

    Start:

      *** START OF THIS PROJECT GUTENBERG EBOOK ... ***

      Some multiple lines of text

    End:

      End of the Project Gutenberg EBook of ...

      *** END OF THIS PROJECT GUTENBERG EBOOK ... ***

    """
    # TODO: There is still some text to be dealt with.
    matches = re.findall(
        (r"^\*\*\* START OF THIS PROJECT.+?$"
         r"(.+?)"  # The text to capture
         r"^\*\*\* END OF THIS PROJECT.+?$"),
        text, flags=re.DOTALL | re.MULTILINE | re.UNICODE)
    return matches[0].strip()


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
    """Get text from Gutenberg based on id

    Parameters
    ----------
    id : int or str
        Identifier.

    Returns
    -------
    text : str or None
        The text. Returns none if the page does not exist.

    """
    url = "http://www.gutenberg.org/ebooks/{id}.txt.utf-8".format(id=id)
    response = requests.get(url)
    return response.content


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    if arguments['get']:
        text = get_text_by_id(arguments['<id>'])
        extracted_text = extract_text(text)
        print(extracted_text)

    elif arguments['list']:
        df = get_list_from_wikidata()
        print(df.to_csv(encoding='utf-8'))


if __name__ == '__main__':
    main()
