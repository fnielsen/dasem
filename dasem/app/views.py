"""Views for dasem app."""


from re import findall, UNICODE

from flask import Blueprint, current_app, render_template, request

from six import u

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


main = Blueprint('main', __name__, template_folder='templates')


@main.route("/")
def index():
    """Return index page of for app."""
    q = request.args.get('q', '')
    words = [word.lower() for word in findall('\w+', q, flags=UNICODE)]

    # Dannet
    if current_app.dasem_dannet is None:
        dannet_synsets_table, dannet_relations_table = None, None
    else:
        dannet_synsets_table, dannet_relations_table = '(empty)', '(empty)'
        if words:
            word = words[0]

            # https://github.com/yhat/db.py/issues/90
            # df = current_app._dasem_dannet.db.query(
            #     query, data={'word': first_word})

            query = DANNET_SYNSET_QUERY.format(word=word)
            dannet_synsets_table = current_app.dasem_dannet.db.query(
                query).to_html()

            query = DANNET_RELATIONS_QUERY.format(word=word)
            dannet_relations_table = current_app.dasem_dannet.db.query(
                query).to_html()

    # FastText
    if current_app.dasem_fast_text is None:
        fast_text_similar = None
    else:
        query = u(" ").join(words)
        fast_text_similar = current_app.dasem_fast_text.most_similar(
            query, top_n=30)

    # Word2Vec
    if current_app.dasem_w2v is None:
        w2v_similar = None
    else:
        try:
            w2v_similar = current_app.dasem_w2v.most_similar(words, top_n=30)
        except (KeyError, ValueError):
            # Word not in vocabulary
            w2v_similar = []

    # Wikipedia ESA
    if current_app.dasem_wikipedia_esa is None:
        esa_related = None
    else:
        esa_related = current_app.dasem_wikipedia_esa.related(q)

    # EParole
    if current_app.dasem_eparole is None:
        eparole_lemmas = None
    else:
        eparole_lemmas = {}
        for word in words:
            eparole_lemmas[word] = current_app.dasem_eparole.word_to_lemmas(
                word)

    return render_template(
        'index.html', q=q,
        dannet_synset_table=dannet_synsets_table,
        dannet_relations_table=dannet_relations_table,
        fast_text_similar=fast_text_similar,
        w2v_similar=w2v_similar,
        esa_related=esa_related,
        eparole_lemmas=eparole_lemmas)
