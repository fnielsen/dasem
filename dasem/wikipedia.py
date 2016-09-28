#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Wikipedia interface.

Usage:
  dasem.wikipedia category-graph | count-category-pages
  dasem.wikipedia count-pages | count-pages-per-user
  dasem.wikipedia article-link-graph [options]
  dasem.wikipedia iter-pages | iter-article-words [options]
  dasem.wikipedia doc-term-matrix [options]
  dasem.wikipedia save-tfidf-vectorizer [options]

Options:
  -h --help            Help
  -v --verbose         Verbose messages
  --display
  --max-n-pages=<int>  Maximum number of pages to iterate over
  --filename=<str>     Filename

"""

from __future__ import division, print_function

from bz2 import BZ2File

import codecs

from collections import Counter

import re

import json

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

from lxml import etree

import mwparserfromhell

from numpy import corrcoef

from scipy.sparse import lil_matrix

from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm


BZ2_XML_DUMP_FILENAME = ('/home/faan/data/wikipedia/'
                         'dawiki-20160901-pages-articles.xml.bz2')

TFIDF_VECTORIZER_FILENAME = 'tfidfvectorizer.json'


def is_article_link(wikilink):
    """Return True is wikilink is an article link.

    Parameters
    ----------
    wikilink : str
        Wikilink to be tested

    Returns
    -------
    result : bool
        True is wikilink is an article link

    Examples
    --------
    >>> is_article_link('[[Danmark]]')
    True
    >>> is_article_link('[[Kategori:Danmark]]')
    False

    """
    if wikilink.startswith('[[') and len(wikilink) > 4:
        wikilink = wikilink[2:]
    if not (wikilink.startswith('Diskussion:')
            or wikilink.startswith('Fil:')
            or wikilink.startswith('File:')
            or wikilink.startswith('Kategori:')
            or wikilink.startswith('Kategoridiskussion:')
            or wikilink.startswith('Wikipedia:')
            or wikilink.startswith('Wikipedia-diskussion:')
            or wikilink.startswith(u'Hjælp:')
            or wikilink.startswith(u'Hjælp-diskussion')
            or wikilink.startswith('Bruger:')
            or wikilink.startswith('Brugerdiskussion:')):
        return True
    return False


def strip_wikilink_to_article(wikilink):
    """Strip wikilink to article.

    Parameters
    ----------
    wikilink : str
        Wikilink

    Returns
    -------
    stripped_wikilink : str
        String with stripped wikilink.

    Examples
    --------
    >>> strip_wikilink_to_article('[[dansk (sprog)|dansk]]')
    'dansk (sprog)'

    >>> strip_wikilink_to_article('Danmark')
    'Danmark'

    """
    if wikilink.startswith('[['):
        wikilink = wikilink[2:-2]
    return wikilink.split('|')[0]


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

    Attributes
    ----------
    file : file
        File object to read from.
    filename : str
        Filename of dump file.
    word_pattern : _sre.SRE_Pattern
        Compile regular expressions for finding words.

    """

    def __init__(self, filename=BZ2_XML_DUMP_FILENAME):
        """Prepare dump file for reading.

        Parameters
        ----------
        filename : str
            Filename or the XML dump file.

        """
        self.filename = filename

        self.word_pattern = re.compile(
            r"""{{.+?}}|
            <!--.+?-->|
            \[\[Fil.+?\]\]|
            \[\[Kategori:.+?\]\]|
            \[http.+?\]|(\w+(?:-\w+)*)""",
            flags=re.UNICODE | re.VERBOSE | re.DOTALL)

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
        if self.filename.endswith('.bz2'):
            self.file = BZ2File(self.filename)
        else:
            self.file = file(self.filename)

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

        Counts for both 'username' and 'ip' are recorded.

        Returns
        -------
        counts : collections.Counter
            Counter object containing counts as values.

        """
        counts = Counter()
        for page in self.iter_pages():
            if 'username' in page:
                counts[page['username']] += 1
            elif 'ip' in page:
                counts[page['ip']] += 1
        return counts

    def iter_article_pages(self, max_n_pages=None, display=False):
        """Iterate over article pages.

        Parameters
        ----------
        max_n_pages : int or None
            Maximum number of pages to return.

        Yields
        ------
        page : dict

        """
        n = 0
        for page in tqdm(self.iter_pages(), disable=not display):
            if page['ns'] == '0':
                n += 1
                yield page
                if max_n_pages is not None and n >= max_n_pages:
                    break

    def iter_article_words(self, max_n_pages=None):
        """Iterate over articles returning word list.

        Parameters
        ----------
        max_n_pages : int or None
            Maximum number of pages to iterate over.

        Yields
        ------
        title : str
            Title of article
        words : list of str
            List of words

        """
        for page in self.iter_article_pages(max_n_pages=max_n_pages):
            words = self.word_pattern.findall(page['text'])
            words = [word.lower() for word in words if word]
            yield page['title'], words

    def article_link_graph(self, verbose=False):
        """Return article link graph.

        Returns
        -------
        graph : dict
            Dictionary with values as a list where elements indicate
            article linked to.

        """
        graph = {}
        for n, page in enumerate(self.iter_article_pages()):
            wikicode = mwparserfromhell.parse(page['text'])
            wikilinks = wikicode.filter_wikilinks()
            article_links = []
            for wikilink in wikilinks:
                if is_article_link(wikilink):
                    article_link = strip_wikilink_to_article(wikilink)
                    article_links.append(article_link.title())
            graph[page['title']] = article_links
            if verbose and not n % 100:
                print(n)
        return graph

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
        """Return category graph.

        Returns
        -------
        graph : dict
            Dictionary with values indicating supercategories.

        """
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

    def doc_term_matrix(self, max_n_pages=None, verbose=False):
        """Return doc-term matrix.

        Parameters
        ----------
        max_n_pages : int or None
            Maximum number of Wikipedia articles to iterate over.
        verbose : bool
            Display message during processing.

        """
        # Identify terms
        n_pages = 0
        all_terms = []
        for title, words in self.iter_article_words(max_n_pages=max_n_pages):
            n_pages += 1
            all_terms.extend(words)
            if verbose and not n_pages % 100:
                print(u"Identified terms from article {}".format(n_pages))
        terms = list(set(all_terms))
        n_terms = len(terms)

        if verbose:
            print("Constructing sparse matrix of size {}x{}".format(
                n_pages, n_terms))
        matrix = lil_matrix((n_pages, n_terms))

        # Count terms wrt. articles
        rows = []
        columns = dict(zip(terms, range(len(terms))))
        for n, (title, words) in enumerate(self.iter_article_words(
                max_n_pages=max_n_pages)):
            rows.append(title)
            for word in words:
                matrix[n, columns[word]] += 1
            if verbose and not n % 100:
                print(u"Sat counts in matrix from article {}".format(n))

        return matrix, rows, terms


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)
    display = arguments['--display']
    if arguments['--max-n-pages'] is None:
        max_n_pages = None
    else:
        max_n_pages = int(arguments['--max-n-pages'])
    verbose = arguments['--verbose']
    
    dump_file = XmlDumpFile()

    if arguments['iter-pages']:
        for page in dump_file.iter_pages():
            print(json.dumps(page))

    elif arguments['count-pages']:
        count = dump_file.count_pages()
        print(count)

    elif arguments['count-pages-per-user']:
        counts = dump_file.count_pages_per_user().most_common(100)
        for n, (user, count) in enumerate(counts, 1):
            print(u"{:4} {:6} {}".format(n, count, user))

    elif arguments['article-link-graph']:
        graph = dump_file.article_link_graph(
            verbose=arguments['--verbose'])
        print(graph)

    elif arguments['category-graph']:
        graph = dump_file.category_graph()
        print(graph)

    elif arguments['count-category-pages']:
        count = dump_file.count_category_pages()
        print(count)

    elif arguments['iter-article-words']:
        for title, words in dump_file.iter_article_words(
                max_n_pages=int(arguments['--max-n-pages'])):
            print(json.dumps([title, words]))

    elif arguments['doc-term-matrix']:
        matrix, rows, columns = dump_file.doc_term_matrix(
            max_n_pages=int(arguments['--max-n-pages']),
            verbose=arguments['--verbose'])
        print(matrix)
        # df = DataFrame(matrix, index=rows, columns=columns)
        # print(df.to_csv(encoding='utf-8'))

    elif arguments['save-tfidf-vectorizer']:
        if arguments['--filename']:
            filename = arguments['--filename']
        else:
            filename = TFIDF_VECTORIZER_FILENAME

        texts = (page['text'] for page in dump_file.iter_article_pages(
            max_n_pages=max_n_pages,
            display=display))

        # Cannot unzip the iterator
        titles = [page['title']
                  for page in dump_file.iter_article_pages(
                          max_n_pages=max_n_pages,
                          display=display)]

        if display:
            tqdm.write('TFIDF vectorizing')
        transformer = TfidfVectorizer()
        transformer.fit(texts)
        transformer.rows = titles

        if display:
            tqdm.write('Writing tfidf vectorizer to {}'.format(filename))
        with codecs.open(filename, 'w', encoding='utf-8') as f:
            f.write(jsonpickle.encode(transformer))

        
if __name__ == '__main__':
    main()
