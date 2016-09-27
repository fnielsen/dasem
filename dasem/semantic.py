"""Semantic.

Usage:
  dasem.semantic relatedness <phrase1> <phrase2>

"""


class Semantic(object):

    def relatedness(self, phrase1, phrase2):
        """Return semantic relatedness between two phrases.

        Parameters
        ----------
        phrase1 : str
            Frist phrase
        phrase2 : str
            Second phrase

        Returns
        -------
        relatedness : float
            Value between 0 and 1 for semantic relatedness

        """
        raise NotImplementedError


def main():
    from docopt import docopt

    arguments = docopt(__doc__)
    phrase1 = arguments['<phrase1>']
    phrase2 = arguments['<phrase2>']

    semantic = Semantic()
    relatedness = semantic.relatedness(phrase1, phrase2)
    print(relatedness)


if __name__ == '__main__':
    main()
