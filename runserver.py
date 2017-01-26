from dasem.app import app

# WSGIServer server better than werkzeug
# http://stackoverflow.com/questions/37962925/
from gevent.wsgi import WSGIServer
http_server = WSGIServer(('', 5000), app)
http_server.serve_forever()
