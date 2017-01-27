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
    
Get first two sentences from Dannet synsets examples:

.. code-block:: bash

    $ python -m dasem.dannet get-all-sentences | head -n 2
    I september måned var jeg sammen med en dansk gruppe af unge bøsser og lesbiske i Moskva
    Til en gruppe på 10 børn i alderen 0-3 år søges pr. 1.3.83 en pædagog 40 timer ugentligt
