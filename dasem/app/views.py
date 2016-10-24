"""Views for dasem app."""


from re import findall, UNICODE

from flask import render_template, request

from . import app


DANNET_SYNSET_QUERY = u"""
select w.form, ws.word_id, s.label, ws.synset_id, w.pos, s.gloss
from words w, wordsenses ws, synsets s
where w.form = '{word}'
  and ws.word_id = w.word_id
  and s.synset_id = ws.synset_id
"""

DANNET_RELATIONS_QUERY = u"""
select s.synset_id, s.label, r.name, r.name2
from (
  select r.value
  from words w, wordsenses ws, relations r
  where w.form = '{word}'
    and w.word_id = ws.word_id
    and r.synset_id = ws.synset_id)
  as a, synsets s, relations r
where s.synset_id = a.value and r.synset_id = s.synset_id;
"""


@app.route("/")
def index():
    """Return index page of for app."""
    q = request.args.get('q', '')
    words = [word.lower() for word in findall('\w+', q, flags=UNICODE)]
    dannet_synsets_table, dannet_relations_table = '(empty)', '(empty)'
    if words:
        word = words[0]

        # https://github.com/yhat/db.py/issues/90
        # df = app._dasem_dannet.db.query(
        #     query, data={'word': first_word})

        query = DANNET_SYNSET_QUERY.format(word=word)
        dannet_synsets_table = app.dasem_dannet.db.query(query).to_html()

        query = DANNET_RELATIONS_QUERY.format(word=word)
        dannet_relations_table = app.dasem_dannet.db.query(query).to_html()

    try:
        w2v_similar = app.dasem_wikipedia_w2v.most_similar(words)
    except (KeyError, ValueError):
        # Word not in vocabulary
        w2v_similar = []
    esa_related = app.dasem_wikipedia_esa.related(q)

    eparole_lemmas = {}
    for word in words:
        eparole_lemmas[word] = app.dasem_eparole.word_to_lemmas(word)

    return render_template(
        'index.html', q=q,
        dannet_synset_table=dannet_synsets_table,
        dannet_relations_table=dannet_relations_table,
        w2v_similar=w2v_similar,
        esa_related=esa_related,
        eparole_lemmas=eparole_lemmas)
