"""Count the words with 'komst'."""

from collections import Counter
from dasem.fullmonty import Fullmonty

corpus = Fullmonty()

komst_words = []
for words in corpus.iter_sentence_words():
    for word in words:
        if 'komst' in word:
            print(word)
            komst_words.append(word.lower())

counts = Counter(komst_words)
print(counts.most_common(200))
