"""Dasem app."""


from __future__ import absolute_import, division, print_function

from flask import Flask
from flask_bootstrap import Bootstrap

from ..dannet import Dannet
from ..eparole import EParole
from ..wikipedia import ExplicitSemanticAnalysis
from ..fullmonty import FastText, Word2Vec


def create_app(enabled_features=('fasttext', 'word2vec',)):
    """Create app.

    Factory for app.

    Parameters
    ----------
    enabled_features : list of str, optional
        Toggle to enable 'word2vec' in app

    """
    app = Flask(__name__)

    Bootstrap(app)

    app.logger.info('Setting up datasets')

    if 'dannet' in enabled_features:
        app.dasem_dannet = Dannet()
    else:
        app.dasem_dannet = None

    if 'esa' in enabled_features:
        app.dasem_wikipedia_esa = ExplicitSemanticAnalysis()
    else:
        app.dasem_wikipedia_esa = None

    if 'fasttext' in enabled_features:
        app.dasem_fast_text = FastText()
    else:
        app.dasem_fast_text = None

    if 'word2vec' in enabled_features:
        app.dasem_w2v = Word2Vec()
    else:
        app.dasem_w2v = None

    if 'eparole' in enabled_features:
        app.dasem_eparole = EParole()
    else:
        app.dasem_eparole = None

    app.logger.info('Datasets loaded')

    from .views import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
