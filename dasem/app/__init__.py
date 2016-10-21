"""Dasem app."""


from __future__ import absolute_import, division, print_function

from flask import Flask
from flask_bootstrap import Bootstrap

from ..dannet import Dannet
from ..wikipedia import ExplicitSemanticAnalysis, Word2Vec


app = Flask(__name__)
Bootstrap(app)

app.dasem_dannet = Dannet()
app.dasem_wikipedia_esa = ExplicitSemanticAnalysis(display=True)
app.dasem_wikipedia_w2v = Word2Vec()

from . import views
