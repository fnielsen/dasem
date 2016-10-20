"""Views for dasem app."""


from re import findall, UNICODE

from flask import render_template, request

from . import app


DANNET_SYNSET_QUERY = u"""
select w.form, ws.word_id, w.pos, s.gloss 
from words w, wordsenses ws, synsets s
where w.form = '{word}'
  and ws.word_id = w.id
  and s.id = ws.synset_id
"""

DANNET_RELATIONS_QUERY = u"""
select s.label 
from (
  select r.value
  from words w, wordsenses ws, relations r
  where w.form = '{word}'
    and w.id = ws.word_id
    and r.synset_id = ws.synset_id)
  as a, synsets s
where s.id = a.value;
"""


@app.route("/")
def index():
    """Return index page of for app."""
    q = request.args.get('q', '')
    first_word = q.strip().split(' ')[0].lower()
    dannet_synsets_table, dannet_relations_table = '(empty)', '(empty)'
    if q:
        first_word = findall('\w+', first_word, flags=UNICODE)[0]

        # https://github.com/yhat/db.py/issues/90
        # df = app._dasem_dannet.db.query(
        #     query, data={'word': first_word})
        
        query = DANNET_SYNSET_QUERY.format(word=first_word)
        dannet_synsets_table = app.dasem_dannet.db.query(query).to_html()

        query = DANNET_RELATIONS_QUERY.format(word=first_word)
        dannet_relations_table = app.dasem_dannet.db.query(query).to_html()

    related = app.dasem_wikipedia_esa.related(q)
        
    return render_template(
        'index.html', q=q,
        dannet_synset_table=dannet_synsets_table,
        dannet_relations_table=dannet_relations_table,
        semantic_related=related)
