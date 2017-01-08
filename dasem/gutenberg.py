"""gutenberg.

Usage:
  dasem.gutenberg list

"""


from __future__ import print_function

from pandas import DataFrame

from sparql import Service


SPARQL_QUERY = """
SELECT ?work ?workLabel ?authorLabel ?gutenberg WHERE {
  ?work wdt:P2034 ?gutenberg.
  ?work wdt:P407 wd:Q9035 .
  OPTIONAL { ?work wdt:P50 ?author . }
  service wikibase:label { bd:serviceParam wikibase:language "da" }
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


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    if arguments['list']:
        df = get_list_from_wikidata()
        print(df.to_csv(encoding='utf-8'))


if __name__ == '__main__':
    main()
