#  -*- coding: utf-8 -*-
# python -m dasem.fullmonty get-all-tokenized-sentences -o tokenized_sentences.txt

from __future__ import print_function

from six import u

from dasem.fullmonty import WordCounts

# CC-BY-SA From Wikipedia: https://da.wikipedia.org/wiki/Forventning
TEXT = """
En forventning er en følelse, der opstår ved usikkerhed. Det er en
tanke centreret om fremtiden og kan være både realistisk og
urealistisk. Et mindre heldigt udfald giver anledning til
skuffelse. Ved en hændelse, der ikke er forventet, fremkommer en
overraskelse. Hvis en forventning indfries vil den typisk resultere i
glæde for personen, der har haft forventningen.  

Glæde og skuffelse kan svinge i styrke efter forventningens
styrke. Forventninger ses primært hos voksne, og virker som om det
udvikles gennem barndommen, små børn forventer intet af verden og
tager den som den kommer, derfor bliver små børn sjældent
skuffede.[Kilde mangler] 

I spilteori bruges forventning også til at forudsige sandsynligheden
af et kommende udfald. Dette bruges inden for mange videnskabelige
grene. Herunder genetikken hvor man kan have en forventning om hvor
mange individer af en given genotype, der vil fremavles i kommende
generation. 
"""

print(TEXT)

try:
    TEXT = TEXT.decode('utf-8')
except:
    pass
 

word_counts = WordCounts()
print(word_counts.extract_keywords(TEXT, top_n=50))
