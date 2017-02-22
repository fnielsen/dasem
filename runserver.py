"""Entrypoint to start app."""

from gevent.wsgi import WSGIServer

import logging

from dasem.app import create_app


app = create_app(logging_level=logging.DEBUG)

# WSGIServer server better than werkzeug
# http://stackoverflow.com/questions/37962925/
http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
