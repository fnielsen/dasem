r"""gutenberg.

Usage:
  dasem.gutenberg --help
  dasem.gutenberg download [options]
  dasem.gutenberg get [options] <id>
  dasem.gutenberg get-all-sentences [options]
  dasem.gutenberg get-all-texts [options]
  dasem.gutenberg get-all-tokenized-sentences [options]
  dasem.gutenberg list-all-ids
  dasem.gutenberg list
  dasem.gutenberg most-similar [options] <word>
  dasem.gutenberg train-and-save-word2vec [options]

Options:
  --debug           Debug information
  --ie=encoding     Input encoding [default: utf-8]
  -o --output=file  Output filename, default output to stdout
  --oe=encoding     Output encoding [default: utf-8]
  -v --verbose      Verbose information

Description:
  This is an interface to Danish texts on Gutenberg.

  There is restriction on how the data should be downloaded from Gutenberg.
  This is stated on their homepage. Download of all the Danish language text
  must be done in the below way.

  wget -w 2 -m -H \
    "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=da"

  The default directory for the data is: `~/dasem_data/gutenberg/`.

  Danish works in Project Gutenberg are to some extent indexed on Wikidata. The
  works can be queried with:

    select ?work ?workLabel where {
      ?work wdt:P2034 ?gutenberg .
      ?work wdt:P364 wd:Q9035 .
      service wikibase:label { bd:serviceParam wikibase:language "da" }
    }

  The `list` command will query Wikidata.

Examples:
  $ python -m dasem.gutenberg get-all-sentences --output sentences.txt

  $ python -m dasem.gutenberg most-similar mand | grep kvinde
  kvinde

References:
  https://www.gutenberg.org/wiki/Gutenberg:Information_About_Robot_Access_to_our_Pages

"""


from __future__ import absolute_import, print_function

import logging

import re

from os import walk, write
from os.path import isfile, join, sep, split

import signal

from six import b, text_type, u

from subprocess import call

from zipfile import ZipFile

import nltk
from nltk.stem.snowball import DanishStemmer
from nltk.tokenize import WordPunctTokenizer

import requests

from . import models
from .config import data_directory
from .corpus import Corpus
from .wikidata import query_to_dataframe


SPARQL_QUERY = """
SELECT ?work ?workLabel ?authorLabel ?gutenberg WHERE {
  ?work wdt:P2034 ?gutenberg.
  ?work wdt:P407 wd:Q9035 .
  OPTIONAL { ?work wdt:P50 ?author . }
  service wikibase:label { bd:serviceParam wikibase:language "da" }
}
"""

DOWNLOAD_URL = ('http://www.gutenberg.org/robot/harvest?'
                'filetypes[]=txt&langs[]=da')


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

    Notes
    -----
    This function attempts to extract the body of the the returned text.
    The full text contains license information and some header information.

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
    df = query_to_dataframe(SPARQL_QUERY)
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


class Gutenberg(Corpus):
    """Gutenberg.

    Interface to Gutenberg.

    The data will be mirrored/downloaded to a directory like:

        ~/dasem_data/gutenberg/www.gutenberg.lib.md.us

    In regard to encoding of the Project Gutenberg texts: For instance,
    10218 is encoded in "ISO Latin-1". This is stated with the line
    "Character set encoding: ISO Latin-1" in the header of the data file.

    Attributes
    ----------
    data_directory : str
        Top directory where the text are mirrored.
    logger : logging.Logger
        Logging object.
    stemmer : object with stem method
        Object with stem method corresponding to
        nltk.stem.snowball.DanishStemmer.
    sentence_tokenizer . object with tokenize method
        Object with tokenize method for tokenizing a text into sentences.
    whitespaces_pattern : regex pattern
        Regular expression pattern.
    word_tokenizer : object with tokenize method
        Object with tokenize method, corresponding to nltk.WordPunctTokenizer.

    """

    def __init__(self):
        """Set up data directory and other attributes."""
        self.logger = logging.getLogger('dasem.gutenberg.Gutenberg')
        self.logger.addHandler(logging.NullHandler())

        self.data_directory = join(data_directory(), 'gutenberg',
                                   'www.gutenberg.lib.md.us')
        self.sentence_tokenizer = nltk.data.load(
            'tokenizers/punkt/danish.pickle')
        self.whitespaces_pattern = re.compile(
            '\s+', flags=re.DOTALL | re.UNICODE)
        self.word_tokenizer = WordPunctTokenizer()
        self.stemmer = DanishStemmer()

    def download(self, redownload=False):
        r"""Download corpus from Gutenberg homepage.

        This method will use the external 'wget' program that is the only
        download method the Project Gutenberg allows. This is explained on
        their homepage. The command is:

        wget -w 2 -m -H \
          "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=da"

        This method will spawn a subprocess. The 'wget' program needs to be
        installed.

        Parameters
        ----------
        redownload : bool, optional
            If True will attempt to download anew. Otherwise, the method
            tests whether a specific file exists on the local data directory.
            If the file exists, then no files are fetch from Gutenberg.

        References
        ----------
        https://www.gutenberg.org/wiki/Gutenberg%3aInformation_About_Robot_Access_to_our_Pages

        """
        test_filename = join(self.data_directory, '1', '0', '2', '1', '10218',
                             '10218-8.zip')
        if not redownload and isfile(test_filename):
            message = 'Not downloading as the file {} exists'
            self.logger.debug(message.format(test_filename))
            return

        directory = split(self.data_directory)[0]
        self.logger.info('Downloading Danish Gutenberg corpus to {}'.format(
            directory))
        call(['wget', '-w', '2', '-m', '-H', DOWNLOAD_URL], cwd=directory)
        self.logger.debug('Gutenberg corpus downloaded')

    def translate_aa(self, text):
        """Translate double-a to 'bolle-aa'.

        Parameters
        ----------
        text : str
            Input text to be translated.

        Returns
        -------
        translated_text : str
            Text with double-a translated to bolle-aa.

        """
        return text.replace(
            'aa', u('\xe5')).replace(
                'Aa', u('\xc5')).replace(
                    'AA', u('\xc5'))

    def translate_whitespaces(self, text):
        r"""Translate multiple whitespaces to a single space.

        Parameters
        ----------
        text : str
            Input string to be translated.

        Returns
        -------
        translated_text : str
            String with multiple whitespaces translated to a single whitespace.

        Examples
        --------
        >>> gutenberg = Gutenberg()
        >>> gutenberg.translate_whitespaces('\n Hello \n  World \n')
        ' Hello World '

        """
        translated_text = self.whitespaces_pattern.sub(' ', text)
        return translated_text

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
        extract_body : bool, default True
            Extract the body of the downloaded/mirrored Gutenberg raw text.

        Returns
        -------
        text : str
            Extracted text. The text is converted to Unicode.

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
        self.logger.debug('Reading text from {}'.format(zip_filename))
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

        if encoded_text.find(b('Character set encoding: ISO-8859-1')) != -1:
            text = encoded_text.decode('ISO-8859-1')
        elif encoded_text.find(b('Character set encoding: ISO Latin-1')) != -1:
            text = encoded_text.decode('Latin-1')
        else:
            raise LookupError('Unknown encoding for file {}'.format(filename))

        if extract_body:
            extracted_text = extract_text(text)
            return extracted_text
        else:
            return text

    def iter_sentence_words(
            self, translate_aa=True, translate_whitespaces=True, lower=True,
            stem=False):
        """Yield list of words from sentences.

        Parameters
        ----------
        translate_aa : bool, default True
            Translate double-a to 'bolle-aa'.
        translate_whitespaces : bool, default True
            Translate multiple whitespaces to single whitespaces
        lower : bool, default True
            Lower case the words.
        stem : bool, default False
            Apply word stemming. DanishStemmer from nltk is used.

        Yields
        ------
        words : list of str
            List of words

        """
        for sentence in self.iter_sentences(
                translate_aa=translate_aa,
                translate_whitespaces=translate_whitespaces):
            words = self.word_tokenizer.tokenize(sentence)
            if lower:
                words = [word.lower() for word in words]
            if stem:
                words = [self.stemmer.stem(word) for word in words]

            yield words

    def iter_sentences(self, translate_aa=True, translate_whitespaces=True):
        """Yield sentences.

        The method uses the NLTK Danish sentence tokenizer.

        Parameters
        ----------
        translate_aa : bool, default True
            Translate double-aa to bolle-aa.
        translate_whitespaces : book, default True
            Translate multiple whitespaces to a single space.

        Yields
        ------
        sentence : str
            String with sentences.

        Examples
        --------
        >>> gutenberg = Gutenberg()
        >>> found = False
        >>> for sentence in gutenberg.iter_sentences():
        ...     if 'Indholdsfortegnelse.' == sentence:
        ...         found = True
        ...         break
        >>> found
        True

        """
        for text in self.iter_texts(translate_aa=translate_aa):
            sentences = self.sentence_tokenizer.tokenize(text)
            for sentence in sentences:
                if translate_whitespaces:
                    sentence = self.translate_whitespaces(sentence)
                yield sentence

    def iter_texts(self, translate_aa=True):
        """Yield texts.

        Parameters
        ----------
        translate_aa : bool, default True
            Translate double-aa to bolle-aa.

        Yields
        ------
        text : str
            Text.

        """
        for id in self.get_all_ids():
            text = self.get_text_by_id(id)
            if translate_aa:
                yield self.translate_aa(text)
            else:
                yield text


class SentenceWordsIterable():
    """Sentence iterable.

    Parameters
    ----------
    translate_aa : bool, default True
        Translate double-a to 'bolle-aa'.
    translate_whitespaces : bool, default True
        Translate multiple whitespaces to single whitespaces
    lower : bool, default True
        Lower case the words.
    stem : bool, default False
        Apply word stemming. DanishStemmer from nltk is used.

    References
    ----------
    https://stackoverflow.com/questions/34166369

    """

    def __init__(self, translate_aa=True, translate_whitespaces=True,
                 lower=True, stem=False):
        """Set up options."""
        self.translate_aa = translate_aa
        self.translate_whitespaces = translate_whitespaces
        self.lower = lower
        self.stem = stem

    def __iter__(self):
        """Restart and return iterable."""
        gutenberg = Gutenberg()
        sentences = gutenberg.iter_sentence_words(
            translate_aa=self.translate_aa,
            translate_whitespaces=self.translate_whitespaces,
            lower=self.lower, stem=self.stem)
        return sentences


class Word2Vec(models.Word2Vec):
    """Gensim Word2vec for Danish Gutenberg corpus.

    Parameters
    ----------
    autosetup : bool, optional
        Determines whether the Word2Vec model should be autoloaded.
    logging_level : logging.ERROR or other, default logging.WARN
        Logging level.

    Notes
    -----
    Trained models can be saved and loaded via the `save` and `load` methods.

    """

    def __init__(self, autosetup=True, logging_level=logging.WARN):
        """Set up model."""
        self.logger = logging.getLogger(__name__ + '.Word2Vec')
        self.logger.addHandler(logging.NullHandler())

        self.model = None
        if autosetup:
            self.logger.info('Autosetup')
            try:
                self.load()
            except:
                self.logger.info('Loading word2vec model failed')
                self.train()
                self.save()

    def data_directory(self):
        """Return data directory.

        Returns
        -------
        directory : str
            Directory as a string.

        """
        directory = join(data_directory(), 'gutenberg')
        return directory

    def full_filename(self, filename):
        """Return filename with full filename path."""
        if sep in filename:
            return filename
        else:
            return join(data_directory(), 'gutenberg', filename)

    def iterable_sentence_words(self, lower=True, stem=False):
        """Return iterable for sentence words.

        Parameters
        ----------
        lower : bool, default True
            Lower case the words.
        stem : bool, default False
            Apply word stemming. DanishStemmer from nltk is used.

        Returns
        -------
        sentence_words : iterable
            Iterable over sentence words

        """
        sentence_words = SentenceWordsIterable(lower=lower, stem=stem)
        return sentence_words


def main():
    """Handle command-line interface."""
    import os

    from docopt import docopt

    arguments = docopt(__doc__)
    if arguments['--output']:
        output_filename = arguments['--output']
        output_file = os.open(output_filename, os.O_RDWR | os.O_CREAT)
    else:
        # stdout
        output_file = 1
    encoding = arguments['--oe']
    input_encoding = arguments['--ie']
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

    # Ignore broken pipe errors
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if arguments['download']:
        gutenberg = Gutenberg()
        gutenberg.download()

    elif arguments['get']:
        gutenberg = Gutenberg()
        text = gutenberg.get_text_by_id(arguments['<id>'])
        write(output_file, text.encode(encoding) + b('\n'))

    elif arguments['get-all-sentences']:
        gutenberg = Gutenberg()
        for sentence in gutenberg.iter_sentences():
            write(output_file, sentence.encode(encoding) + b('\n'))

    elif arguments['get-all-texts']:
        gutenberg = Gutenberg()
        for text in gutenberg.iter_texts():
            write(output_file, text.encode(encoding) + b('\n'))

    elif arguments['get-all-tokenized-sentences']:
        gutenberg = Gutenberg()
        for sentence in gutenberg.iter_tokenized_sentences():
            write(output_file, sentence.encode(encoding) + b('\n'))

    elif arguments['list-all-ids']:
        gutenberg = Gutenberg()
        ids = gutenberg.get_all_ids()
        for id in ids:
            write(output_file, u(id).encode(encoding) + b('\n'))

    elif arguments['list']:
        df = get_list_from_wikidata()
        write(output_file, df.to_csv(encoding=encoding))

    elif arguments['most-similar']:
        # Input argument
        word = arguments['<word>']
        if not isinstance(word, text_type):
            word = word.decode(input_encoding).lower()
        word = word.lower()

        # Compute similarity with the Word2Vec model
        word2vec = Word2Vec()
        words_and_similarity = word2vec.most_similar(word)

        # Output the result
        for word, similarity in words_and_similarity:
            write(output_file, word.encode(encoding) + b('\n'))

    elif arguments['train-and-save-word2vec']:
        word2vec = Word2Vec(autosetup=False)
        word2vec.train()
        logger.info('Saving word2vec model')
        word2vec.save()


if __name__ == '__main__':
    main()
