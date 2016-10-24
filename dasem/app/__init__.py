"""Dasem app."""


from __future__ import absolute_import, division, print_function

import logging

from flask import Flask
from flask_bootstrap import Bootstrap

from ..dannet import Dannet
from ..eparole import EParole
from ..wikipedia import ExplicitSemanticAnalysis, Word2Vec


app = Flask(__name__)
Bootstrap(app)

logging_level = logging.DEBUG
if not app.debug:
    app.logger.setLevel(logging_level)
    logging.basicConfig()

app.logger.info('Setting up datasets')
app.dasem_dannet = Dannet(logging_level=logging_level)
app.dasem_wikipedia_esa = ExplicitSemanticAnalysis(logging_level=logging_level)
app.dasem_wikipedia_w2v = Word2Vec(logging_level=logging_level)
app.dasem_eparole = EParole(logging_level=logging_level)
app.logger.info('Datasets loaded')

from . import views
