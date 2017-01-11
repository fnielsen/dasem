Dasem
=====

Danish semantic analysis.


Examples
--------

Get nouns from Dannet and Wiktionary:

.. code-block:: python

		from dasem.wiktionary import get_nouns
		from dasem.dannet import Dannet

		wiktionary_nouns = get_nouns()

		dannet = Dannet()
		query = "select w.form from words w where w.pos = 'Noun'"
		dannet_nouns = set(dannet.db.query(query).form)

		nouns = dannet_nouns.union(wiktionary_nouns)

Get similar words based on a word2vec model on the Danish part of the Project Gutenberg corpus:

.. code-block:: bash

    $ python -m dasem.gutenberg most-similar mand
    kvinde
    dame
    pige
    kone
    fyr
    dreng
    præst
    profet
    hund
    person
