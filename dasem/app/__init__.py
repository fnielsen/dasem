"""Dasem app."""


from __future__ import absolute_import, division, print_function

from flask import Flask
from flask_bootstrap import Bootstrap

from ..dannet import Dannet
from ..semantic import Semantic


app = Flask(__name__)
Bootstrap(app)

app.dasem_dannet = Dannet()
app.dasem_semantic = Semantic()

from . import views
