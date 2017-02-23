"""Entrypoint to start app."""

from gevent.wsgi import WSGIServer

import logging

from dasem.app import create_app


logging_level = logging.DEBUG

logger = logging.getLogger()
logger.setLevel(logging_level)
logging_handler = logging.StreamHandler()
logging_handler.setLevel(logging_level)
logging_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging_handler.setFormatter(logging_formatter)
logger.addHandler(logging_handler)
logger.info('Logging setup')

logger.info('Creating app')
app = create_app()

# WSGIServer server better than werkzeug
# http://stackoverflow.com/questions/37962925/
http_server = WSGIServer(('', 5000), app, log=logger)
http_server.serve_forever()
