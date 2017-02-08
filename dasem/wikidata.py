"""wikidata.

Usage:
  dasem.wikidata --help
  dasem.wikidata query [options] <query>

Options:
  -h --help  Help message

Examples:
  $ python -m dasem.wikidata query "select * where {?p wdt:P27 wd:Q35} limit 3"
  ,p
  0,http://www.wikidata.org/entity/Q498
  1,http://www.wikidata.org/entity/Q2330
  2,http://www.wikidata.org/entity/Q5015

"""


from __future__ import absolute_import, division, print_function

from pandas import DataFrame

import requests

WIKIDATA_SPARQL_ENDPOINT = 'https://query.wikidata.org/sparql'


def query_to_dataframe(query):
    """Query Wikidata SPARQL and return dataframe.

    Parameters
    ----------
    query : str
        SPARQL query as string.

    Returns
    -------
    df : pandas.DataFrame
       Pandas data frame with response.

    """
    response = requests.get(
        WIKIDATA_SPARQL_ENDPOINT,
        params={'format': 'json', 'query': query})
    data = response.json()['results']['bindings']
    df = DataFrame([{k: v['value'] for k, v in row.items()} for row in data])
    return df


def main():
    """Handle command-line input."""
    from docopt import docopt

    arguments = docopt(__doc__)

    if arguments['query']:
        query = arguments['<query>']
        df = query_to_dataframe(query)
        print(df.to_csv())


if __name__ == '__main__':
    main()
