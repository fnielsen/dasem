r"""retsinformationdk.

Usage:
  dasem.retsinformationdk download [options]
  dasem.retsinformationdk get-all-sentences [options]
  dasem.retsinformationdk get-all-texts [options]
  dasem.retsinformationdk save-gigaword [options]

Options:
  --debug             Debug messages.
  -h --help           Help message
  --oe=encoding       Output encoding [default: utf-8]
  -o --output=<file>  Output filename, default output to stdout
  --separator=<sep>   Separator [default: \n]
  --verbose           Verbose messages.

Notes
-----
The `wget' program is required to download the pages from retsinformation.dk.

"""

import codecs

from datetime import datetime

import json

import logging

import os
from os import listdir, write
from os.path import isfile, join

import re

import signal

from six import u

from subprocess import call

import nltk

from lxml import etree

import pytz

from . import utils


DOWNLOAD_URL = "https://www.retsinformation.dk"

PUBLICATION_DATE_PATTERN = re.compile(
    u(r"Offentligg\u00f8relsesdato: (\d{1,2})-(\d{1,2})-(\d{4})<"),
    flags=re.UNICODE)

# For instance, R0710.aspx?id=10003
HTML_FILE_IDENTIFIER_PATTERN = re.compile(r"R0710\.aspx\?id\=(\d+)")


def data_directory():
    """Return diretory where data should be.

    Returns
    -------
    directory : str
        Data directory.

    """
    directory = join(utils.data_directory(), 'retsinformationdk')
    return directory


def download(redownload=False):
    """Download webpages of retsinformation.dk.

    Parameters
    ----------
    redownload : bool, optional
       Controls whether the webpages should be redownloaded.

    Notes
    -----
    This function uses the `wget` program, so it will need to be installed.

    Download may take considerable time. There is a wait of 5 seconds between
    requests.

    PDF and print pages are left out, e.g.,
    https://www.retsinformation.dk/print.aspx?id=206363
    https://www.retsinformation.dk/pdfPrint.aspx?id=206363

    There seems to be lots of pages where reporting "Last-modified header
    missing" which means the page is downloaded a new.

    """
    logger = logging.getLogger(__name__)

    make_data_directory()

    test_filename = join(
        data_directory(), 'www.retsinformation.dk', 'Forms',
        'R0710.aspx?id=207290')
    if not redownload and isfile(test_filename):
        message = 'Not downloading as the file {} exists'
        logger.debug(message.format(test_filename))
        return

    directory = data_directory()
    logger.info('Downloading Retsinformation.dk corpus to {}'.format(
        directory))
    call(['wget',
          '-w', '5',  # Wait five seconds
          '--recursive',
          '-l', 'inf',
          # '--no-clobber',
          '--timestamping',
          '--exclude-directories', '/includes,/js',
          '--reject-regex', '"(print)|(pdfPrint)"',
          DOWNLOAD_URL],
         cwd=directory)
    logger.debug('Retsinformation.dk corpus downloaded')


def iter_htmls(return_filename=False, return_url=False):
    """Yield HTML strings from retsinformation.dk.

    Parameters
    ----------
    return_filename : bool, optional
        Determine whether the output should include the filename of the read
        file.
    return_url : bool, optional
        Determine whether the output should include the URL of the read
        file.

    Yields
    ------
    text : string
        String with HTML.
    filename : string
        Filename. Only returned if the `return_filename` is True.
    url : string
        Reconstructed URL. Only returned if the `return_url` is True.

    Notes
    -----
    Only content pages of retsinformation.dk will be yielded. These pages are
    identified with the pattern "?id=" and not containing "&rg=".

    """
    directory = join(data_directory(), 'www.retsinformation.dk', 'Forms')
    filenames = listdir(directory)
    for filename in filenames:
        if "?id=" in filename and "&rg=" not in filename:
            full_filename = join(directory, filename)
            url = "https://www.retsinformation.dk/Forms/" + filename
            html = open(full_filename).read()
            if return_filename:
                if return_url:
                    yield html, filename, url
                else:
                    yield html, filename
            else:
                if return_url:
                    yield html, url
                else:
                    yield html


def iter_sentences():
    """Yield sentences from retsinformation.dk.

    Yields
    ------
    sentence : string
        String with sentence.

    Notes
    -----
    For laws a paragraph seems to be identifiable with the XPATH specification
    `//p[@class='Paragraf']`.

    Others (older?) may be identified by `//p[@class='Paragraftekst']`.

    There are also other content pages. "Kendelser", see, e.g.,
    `https://www.retsinformation.dk/Forms/R0710.aspx?id=146212`. These seem to
    be identifiable as `<p class="TekstV">`.

    Some webpages has little formatting, e.g.,
    `https://www.retsinformation.dk/Forms/R0710.aspx?id=206828`. Such webpages
    are ignored for now.

    """
    logger = logging.getLogger(__name__)

    sentence_tokenizer = nltk.data.load('tokenizers/punkt/danish.pickle')

    paragraph1_xpath = "//*[@id='ctl00_MainContent_Broedtekst1']/p"
    paragraph2_xpath = '//div[@id="INDHOLD"]/p'
    paragraph3_xpath = "//div[@class='PARAGRAF']"
    text_xpath = "//p[@class='TekstV']"

    whitespace_pattern = re.compile(r'\s+', re.UNICODE)
    ignore_pattern = re.compile(u(r"""
    ^\s*(?:
    (?:§\s\d+\s*\.?(?:\s.\.)?)
    |
    (?:Stk\.\s*\d+\s*\.?)
    |
    (?:\d+(?:-\d+)?\.)
    |
    (?:\d+(?:\.\d+)+\.?)
    |
    (?:Artikel [XVI]+)
    )\s*$"""), re.UNICODE | re.VERBOSE)

    for html, filename in iter_htmls(return_filename=True):
        tree = etree.HTML(html)

        elements = tree.xpath(paragraph1_xpath)
        if not elements:
            elements = tree.xpath(paragraph2_xpath)
            if not elements:
                elements = tree.xpath(paragraph3_xpath)
                if not elements:
                    elements = tree.xpath(text_xpath)

        if elements:
            for element in elements:
                text = " ".join(element.itertext())
                text = whitespace_pattern.sub(' ', text)
                sentences = sentence_tokenizer.tokenize(text)
                for sentence in sentences:
                    # Here text that are only upper case is ignore.
                    if (not ignore_pattern.match(sentence) and
                            not sentence.isupper()):
                        yield sentence.strip()
        else:
            logger.info('Text not matched in {}'.format(filename))


def iter_texts():
    """Yield texts from retsinformation.dk.

    Yields
    ------
    text : string
        String with text.

    Notes
    -----
    This function will iterate over content pages and extract the body text
    of each content pages. The body text is identified from the HTML tag
    '<div id="ctl00_MainContent_Broedtekst1" retsinformationversion="true">'.

    """
    logger = logging.getLogger(__name__)

    bodytext_xpath = "//div[@id='ctl00_MainContent_Broedtekst1']"

    for html, filename in iter_htmls(return_filename=True):
        tree = etree.HTML(html)
        bodytext_elements = tree.xpath(bodytext_xpath)

        # There are apparently some (or at least one) webpages that
        # follows a URL pattern for content pages but where the page
        # is a list page. For instance,
        # https://www.retsinformation.dk/Forms/R0910.aspx?id=208123
        if len(bodytext_elements) == 1:
            yield " ".join(bodytext_elements[0].itertext())
        else:
            logger.warn('Something is unusual with the file: {}'.format(
                filename))


def make_data_directory():
    """Make data directory for retsinformationdk."""
    utils.make_data_directory(data_directory())


def save_gigaword(directory=None):
    """Extract text and save to files in the gigaword format.

    Extract the text from the HTML downloaded from the retsinformation.dk
    website and save the text in the gigaword format together with a
    metadata file.

    Parameters
    ----------
    directory : str or None
        Directory where ther Gigaword data is stored.

    """
    logger = logging.getLogger(__name__)

    json_filename = "retsinformationdk.jsonl"

    bodytext_xpath = "//div[@id='ctl00_MainContent_Broedtekst1']"

    if directory is None:
        directory = join(data_directory(), "retsinformationdk")
        utils.make_data_directory(directory)

    json_full_filename = join(directory, json_filename)

    timezone = pytz.FixedOffset(60)

    with open(json_full_filename, 'w') as json_fid:
        for file_index, (html, filename, url) in enumerate(
                iter_htmls(return_filename=True,
                           return_url=True)):
            tree = etree.HTML(html)
            bodytext_elements = tree.xpath(bodytext_xpath)

            # There are apparently some (or at least one) webpages that
            # follows a URL pattern for content pages but where the page
            # is a list page. For instance,
            # https://www.retsinformation.dk/Forms/R0910.aspx?id=208123
            if len(bodytext_elements) == 1:
                text = " ".join(bodytext_elements[0].itertext())
            else:
                logger.warn('Something is unusual with the file: {}'.format(
                    filename))

            # Publication date extraction
            matches = PUBLICATION_DATE_PATTERN.findall(html)
            if len(matches) == 1 and len(matches[0]) == 3:
                date_published = datetime(
                    int(matches[0][2]),
                    int(matches[0][1]),
                    int(matches[0][0]), tzinfo=timezone).strftime("%c %z")
            else:
                logger.error("Date could not be extracted: {}".format(
                    filename))

            text = text.strip()
            text = "\n".join(line.strip() for line in text.split('\n'))
            text = re.sub(r"\n\n(?:(\n)\n*)?", r"\n\1", text, flags=re.DOTALL)

            # Document identifier extraction
            matches = HTML_FILE_IDENTIFIER_PATTERN.findall(filename)
            if len(matches) == 1:
                doc_id = "retsinformationdk_" + matches[0]
            else:
                logger.error(
                    "Document identifer could not be extracted: {}".format(
                        filename))

            # Write line in metadata file
            metadata = {
                'uri': url,
                'doc_id': doc_id,
                'date_published': date_published,
                'date_built': datetime.now(timezone).strftime("%c %z"),
            }
            json_fid.write(json.dumps(metadata) + "\n")

            # Write text file
            text_filename = join(directory, doc_id)
            with codecs.open(text_filename, "w", encoding="utf-8") as text_fid:
                text_fid.write(text)

            logger.info("Processed {} [{}]".format(
                filename, file_index))


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    logging_level = logging.WARN
    if arguments['--debug']:
        logging_level = logging.DEBUG
    elif arguments['--verbose']:
        logging_level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(logging_level)
    logging_handler = logging.StreamHandler()
    logging_handler.setLevel(logging_level)
    logging_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging_handler.setFormatter(logging_formatter)
    logger.addHandler(logging_handler)

    if arguments['--output']:
        output_filename = arguments['--output']
        output_file = os.open(output_filename, os.O_RDWR | os.O_CREAT)
    else:
        # stdout
        output_file = 1
    output_encoding = arguments['--oe']
    separator = u(arguments['--separator'])
    if separator == r'\n':
        separator = u('\n')

    # Ignore broken pipe errors
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if arguments['download']:
        download(redownload=True)

    elif arguments['get-all-sentences']:
        for sentence in iter_sentences():
            write(output_file,
                  sentence.encode(output_encoding) +
                  separator.encode(output_encoding))

    elif arguments['get-all-texts']:
        for text in iter_texts():
            write(output_file,
                  text.encode(output_encoding) +
                  separator.encode(output_encoding))

    elif arguments['save-gigaword']:
        save_gigaword()


if __name__ == '__main__':
    main()
