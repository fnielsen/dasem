"""text.

Usage:
  text decompound <text>

"""

from codecs import open

from os.path import join, split

from nltk import sent_tokenize, WordPunctTokenizer
from nltk import word_tokenize as nltk_word_tokenize


class Decompounder(object):
    """Word decompunder."""

    def __init__(self):
        """Set up map."""
        self.word_tokenizer = WordPunctTokenizer()

        filename = join(split(__file__)[0], 'data', 'compounds.txt')

        self.decompound_map = {}
        with open(filename, encoding='utf-8') as fid:
            for line in fid:
                parts = line.strip().split('|')
                compound = "".join(parts)
                decompounded_parts = [part for part in parts
                                      if part != 's' and part != 'e']
                decompounded = " ".join(decompounded_parts)
                self.decompound_map[compound] = decompounded

    def decompound_text(self, text):
        """Return decompounded text.

        Parameters
        ----------
        text : str
            Text as a (unicode) str.

        Returns
        -------
        decompounded : str
            String with decompounded parts separated by a whitespace.

        Examples
        --------
        >>> decompounder = Decompounder()
        >>> text = 'Det er en investeringsvirksomhed'
        >>> decomp = decompounder.decompound_text(text)
        >>> decomp == 'det er en investering virksomhed'
        True

        """
        tokens = self.word_tokenizer.tokenize(text)
        return " ".join(self.decompound_word(token.lower())
                        for token in tokens)

    def decompound_word(self, word):
        """Return decompounded word.

        Parameters
        ----------
        word : str
            Word as a (unicode) str.

        Returns
        -------
        decompounded : str
            String with decompounded parts separated by a whitespace.

        Examples
        --------
        >>> decompounder = Decompounder()
        >>> decomp = decompounder.decompound_word('investeringsvirksomhed')
        >>> decomp == 'investering virksomhed'
        True

        """
        return self.decompound_map.get(word, word)


def sentence_tokenize(text):
    """Tokenize a Danish text into sentence.

    The model from NTLK trained on Danish is used.

    Parameters
    ----------
    text : str
        The text to be tokenized.

    Returns
    -------
    sentences : list of str
        Sentences as list of strings.

    Examples
    --------
    >>> text = 'Hvad!? Hvor har du f.eks. siddet?'
    >>> sentences = sentence_tokenize(text)
    >>> sentences
    ['Hvad!?', 'Hvor har du f.eks. siddet?']

    """
    return sent_tokenize(text, language='danish')


def word_tokenize(sentence):
    """Tokenize a Danish sentence into words."""
    return nltk_word_tokenize(sentence)


def main():
    """Handle command-line interface."""
    from docopt import docopt

    arguments = docopt(__doc__)

    if arguments['decompound']:
        text = arguments['<text>']
        decompounder = Decompounder()
        decompounded = decompounder.decompound_text(text)
        print(decompounded)

    elif arguments['decompound-word']:
        word = arguments['<word>']
        decompounder = Decompounder()
        decompounded = decompounder.decompound_word(word)
        print(decompounded)

    else:
        assert False


if __name__ == '__main__':
    main()
