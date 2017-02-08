"""adl - Arkiv for dansk litteratur.

Usage:
  dasem.adl get-author-data [options]

Options:
  --debug             Debug messages.
  -h --help           Help message
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  --verbose           Verbose messages.

Description:
  adl is Arkiv for dansk litteratur, a Danish digital archive with old
  literature.

  An example of a URL for download of the raw text to a complete work is:

  http://adl.dk/adl_pub/pg/cv/AsciiPgVaerk2.xsql?p_udg_id=50&p_vaerk_id=554

  which is the novel "Phantasterne". The following is "Bispen paa
  Boerglum og hans Fraende":

  http://adl.dk/adl_pub/pg/cv/AsciiPgVaerk2.xsql?p_udg_id=97&p_vaerk_id=9138

Examples:
  $ python -m dasem.adl get-author-data | head -n 3 | cut -f1-3 -d,
  ,author,author_label
  0,http://www.wikidata.org/entity/Q439370,Herman Bang
  1,http://www.wikidata.org/entity/Q347482,Hans Egede Schack

"""


from __future__ import absolute_import, division, print_function

import os
from os import write

from .wikidata import query_to_dataframe


WIKIDATA_AUTHOR_QUERY = """
select * where {
  ?author wdt:P31 wd:Q5 .
  ?author wdt:P973 ?url .
  filter strstarts(lcase(str(?url)), 'http://adl.dk')
  optional {
    ?author wdt:P21 ?gender .
    ?gender rdfs:label ?gender_label . filter (lang(?gender_label) = 'da')
  }
  ?author rdfs:label ?author_label . filter (lang(?author_label) = 'da')
}
order by ?url
"""


def get_author_data():
    """Return ADL author data from Wikidata.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with author data, including name and gender.

    """
    df = query_to_dataframe(WIKIDATA_AUTHOR_QUERY)
    return df


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    if arguments['--output']:
        output_filename = arguments['--output']
        output_file = os.open(output_filename, os.O_RDWR | os.O_CREAT)
    else:
        # stdout
        output_file = 1
    output_encoding = arguments['--oe']

    if arguments['get-author-data']:
        df = get_author_data()
        write(output_file, df.to_csv(encoding=output_encoding))


if __name__ == '__main__':
    main()
