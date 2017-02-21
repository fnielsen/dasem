"""Plot fasttext embedding."""

from matplotlib.pyplot import show, title

import seaborn as sns

from pandas import DataFrame

from six import u

from dasem.dannet import FastText


fast_text = FastText()

words = ['virksomhed', 'virxsomhed', 'forretning', 'kapital', 'finans',
         'grus', 'grusgrav', 'transportvirksomhed', u('hyrek\xf8rsel')]
vector = fast_text.word_vector(words[0])

df = DataFrame([], index=range(len(vector)), columns=words)

for word in words:
    df.ix[:, word] = fast_text.word_vector(word)

corr = df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
title('Correlation in word embedding')
show()
