"""wikisource.

Usage:
  dasem.wikisource get <title>
  dasem.wikisource list

Example:
  $ python -m dasem.wikisource get Mogens

"""


from __future__ import print_function

import re

from bs4 import BeautifulSoup

import requests

from six import u

from .wikidata import query_to_dataframe


SPARQL_QUERY = """
SELECT distinct ?item ?itemLabel ?article WHERE {
  ?article schema:about ?item.
  ?article schema:isPartOf <https://da.wikisource.org/>.
  values ?kind { wd:Q7725634 wd:Q1372064 wd:Q7366 wd:Q49848}
  ?item (wdt:P31/wdt:P279*) ?kind .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "da,en". }
}
"""


def extract_text(text):
    """Extract relevant part of text from page.

    Attempts with various regular expressions to extract the relevant
    text from the downloaded parsed wikipage.

    Poems might have the '<poem>...</poem>' construct. Text between these two
    tags are extracted and returned.

    Public domain license information is ignored.

    Parameters
    ----------
    text : str
        Downloaded text.

    Returns
    -------
    extracted_text : str
        Extracted text.

    """
    # Match <poem> and just extract that.
    in_poem = re.findall(r'<poem>(.*?)</poem>', text,
                         flags=re.UNICODE | re.DOTALL)
    if in_poem:
        return u"\n\n".join(in_poem)

    # Ignore license information. This might be above or below the text.
    text = re.sub((r'Public domainPublic domain(.*?), '
                   'da det blev udgivet.{15,25}\.$'), '\n',
                  text, flags=re.UNICODE | re.DOTALL | re.MULTILINE)

    regex = r'Teksten\[redig' + u('\xe9') + r'r\](.*)'
    after_teksten = re.findall(regex, text, flags=re.UNICODE | re.DOTALL)
    if after_teksten:
        return u"\n\n".join(after_teksten)

    # Match bottom of infobox on some of the songs
    rest = re.findall(r'.*Wikipedia-link\s*(.*)', text,
                      flags=re.UNICODE | re.DOTALL)
    if rest:
        return u"\n\n".join(rest)

    return text


def get_list_from_wikidata():
    """Get list of works from Wikidata.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with information from Wikidata.

    """
    df = query_to_dataframe(SPARQL_QUERY)
    return df


def get_text_by_title(title):
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
        text = BeautifulSoup(data['parse']['text']['*'], "lxml").get_text()
    else:
        text = None
    return text


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    if arguments['get']:
        text = get_text_by_title(arguments['<title>'])
        if text:
            extracted_text = extract_text(text)
            print(extracted_text.encode('utf-8'))

    elif arguments['list']:
        df = get_list_from_wikidata()
        print(df.to_csv(encoding='utf-8'))


if __name__ == '__main__':
    main()
