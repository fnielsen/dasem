"""wikisource.

Usage:
  dasem.wikisource get <title>
  dasem.wikisource list

Example:
  $ python -m dasem.wikisource get Mogens

"""


from __future__ import print_function

from bs4 import BeautifulSoup

import requests

from pandas import DataFrame

from sparql import Service


SPARQL_QUERY = """
SELECT ?item ?itemLabel ?article WHERE {
  ?article schema:about ?item.
  ?article schema:isPartOf <https://da.wikisource.org/>.
  values ?kind { wd:Q7725634 wd:Q1372064 wd:Q7366 }
  ?item (wdt:P31/wdt:P291*) ?kind .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "da,en". }
}
"""


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


def get_text_from_title(title):
    """Get text from Wikisource based on title.

    If the text is split over several wikipages (which is the case with novels)
    then the full text will not be returned, - only the index page.

    Parameters
    ----------
    title : str
        Title of wikipage on Danish Wikisource.

    Returns
    -------
    text : str or None
        The text. Returns none if the page does not exist.

    """
    url = 'https://da.wikisource.org/w/api.php'
    params = {'page': title, 'action': 'parse', 'format': 'json'}
    data = requests.get(url, params=params).json()
    if 'parse' in data:
        text = BeautifulSoup(data['parse']['text']['*']).get_text()
    else:
        text = None
    return text


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    if arguments['get']:
        text = get_text_from_title(arguments['<title>'])
        print(text.encode('utf-8'))

    elif arguments['list']:
        df = get_list_from_wikidata()
        print(df.to_csv(encoding='utf-8'))


if __name__ == '__main__':
    main()
