"""Dasem app."""


from __future__ import absolute_import, division, print_function

from flask import Flask
from flask_bootstrap import Bootstrap

from ..dannet import Dannet


app = Flask(__name__)
Bootstrap(app)

app._dasem_dannet = Dannet()

from . import views
