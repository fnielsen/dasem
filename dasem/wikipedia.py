"""Wikipedia interface.

Usage:
  wikipedia.py category_graph | count_category_pages
  wikipedia.py count_pages | count_pages_per_user

"""

from __future__ import division, print_function

from bz2 import BZ2File

from collections import Counter

from lxml import etree

import mwparserfromhell


BZ2_XML_DUMP_FILENAME = ('/home/faan/data/wikipedia/'
                         'dawiki-20160901-pages-articles.xml.bz2')


def strip_to_category(category):
    """Strip prefix and postfix from category link.

    Parameters
    ----------
    category : str

    Returns
    -------
    stripped_category : str
        String with stripped category

    """
    if category.startswith('[[Kategori:'):
        category = category[11:-2]
    elif category.startswith('Kategori:'):
        category = category[9:]
    return category.split('|')[0]


class XmlDumpFile(object):
    """XML Dump file.

    For instance, dawiki-20160901-pages-articles.xml.bz2.

    """

    def __init__(self, filename=BZ2_XML_DUMP_FILENAME):
        """Prepare dump file for reading.

        Parameters
        ----------
        filename : str
            Filename or the XML dump file.

        """
        self.filename = filename

        if filename.endswith('.bz2'):
            self.file = BZ2File(filename)
        else:
            self.file = file(filename)

    def clean_tag(self, tag):
        """Remove namespace from tag.

        Parameters
        ----------
        tag : str
            Tag with namespace prefix.

        Returns
        -------
        cleaned_tag : str
            Tag with namespace part removed.

        """
        cleaned_tag = tag.split('}')[-1]
        return cleaned_tag

    def iter_elements(self, events=('end',)):
        """Iterate over elements in XML file.

        Yields
        ------
        event : str
            'start' or 'end'
        element : Element
            XML element

        """
        with self.file as f:
            for event, element in etree.iterparse(f, events=events):
                yield event, element

    def iter_page_elements(self, events=('end',)):
        """Iterator for page XML elements."""
        for event, element in self.iter_elements(events=events):
            tag = self.clean_tag(element.tag)
            if tag == 'page':
                yield event, element

    def iter_pages(self):
        """Iterator for page yielding a dictionary.

        Yields
        ------
        page : dict

        """
        for event, element in self.iter_page_elements(events=('end',)):
            page = {}
            for descendant in element.iterdescendants():
                tag = self.clean_tag(descendant.tag)
                if tag not in ['contributor', 'revision']:
                    page[tag] = descendant.text
            yield page

    def count_pages(self):
        """Return number of pages.

        Returns
        -------
        count : int
            Number of pages

        """
        count = 0
        for event, element in self.iter_page_elements():
            count += 1
        return count

    def count_pages_per_user(self):
        """Count the number of pages per user.

        Returns
        -------
        counts : collections.Counter
            Counter object containing counts as values.

        """
        counts = Counter()  # defaultdict(int)
        for page in self.iter_pages():
            if 'username' in page:
                counts[page['username']] += 1
            elif 'ip' in page:
                counts[page['ip']] += 1
        return counts

    def iter_category_pages(self):
        """Iterate over category pages.

        For dawiki-20160901-pages-articles.xml.bz2 this method
        returns 51548

        Yields
        ------
        page : dict

        """
        for page in self.iter_pages():
            if page['ns'] == '14':
                yield page

    def count_category_pages(self):
        """Count category pages.

        Returns
        -------
        count : int
            Number of category pages.

        """
        n = 0
        for page in self.iter_category_pages():
            n += 1
        return n

    def category_graph(self):
        """Return category graph."""
        graph = {}
        for page in self.iter_category_pages():
            wikicode = mwparserfromhell.parse(page['text'])
            wikilinks = wikicode.filter_wikilinks()
            categories = []
            for wikilink in wikilinks:
                if wikilink.startswith('[[Kategori:'):
                    categories.append(strip_to_category(wikilink))
            category = strip_to_category(page['title'])
            graph[category] = categories
        return graph


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    dump_file = XmlDumpFile()

    if arguments['category_graph']:
        graph = dump_file.category_graph()
        print(graph)

    elif arguments['count_category_pages']:
        count = dump_file.count_category_pages()
        print(count)

    elif arguments['count_pages']:
        count = dump_file.count_pages()
        print(count)

    elif arguments['count_pages_per_user']:
        counts = dump_file.count_pages_per_user().most_common(100)
        for n, (user, count) in enumerate(counts, 1):
            print(u"{:4} {:6} {}".format(n, count, user))


if __name__ == '__main__':
    main()
