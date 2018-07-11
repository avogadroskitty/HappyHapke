from __future__ import division, print_function
import io
import json
import tornado.ioloop
import tornado.web
import tornado.websocket
from matplotlib.backends.backend_webagg_core import FigureManagerWebAgg


class MplWebApp(tornado.web.Application):
  def __init__(self, routes, **kwargs):
    # routes common to all webagg servers
    mplweb_routes = [
        (r'/([0-9]+)/download.([a-z0-9.]+)', DownloadHandler),
        (r'/([0-9a-f]+)/([0-9]+)/ws', WebSocketHandler),
        (r'/mpl.js', MplJsHandler),
        (r'/_static/(.*)', tornado.web.StaticFileHandler,
         dict(path=FigureManagerWebAgg.get_static_file_path())),
    ]
    tornado.web.Application.__init__(self, routes + mplweb_routes, **kwargs)
    self.prog_states = {}  # uid -> ProgramState
    self.fig_managers = {}  # fignum -> manager
    # hack in a mock manager for keep-alive sockets
    self.fig_managers['0'] = _MockFigureManager()


class MplJsHandler(tornado.web.RequestHandler):
  def get(self):
    self.set_header('Content-Type', 'application/javascript')
    self.write(FigureManagerWebAgg.get_javascript())


class DownloadHandler(tornado.web.RequestHandler):
  def get(self, fignum, fmt):
    mimetypes = {
        'ps': 'application/postscript',
        'eps': 'application/postscript',
        'pdf': 'application/pdf',
        'svg': 'image/svg+xml',
        'png': 'image/png',
        'jpeg': 'image/jpeg',
        'tif': 'image/tiff',
        'emf': 'application/emf'
    }
    self.set_header('Content-Type', mimetypes.get(fmt, 'binary'))
    buff = io.BytesIO()
    self.application.fig_managers[fignum].canvas.print_figure(buff, format=fmt)
    self.write(buff.getvalue())


class WebSocketHandler(tornado.websocket.WebSocketHandler):
  supports_binary = True

  def open(self, uid, fignum):
    self.uid = uid
    self.fignum = fignum
    self.application.fig_managers[fignum].add_web_socket(self)
    if hasattr(self, 'set_nodelay'):
      self.set_nodelay(True)

  def on_close(self):
    app = self.application
    print('closing ws', self.uid, self.fignum)
    app.fig_managers[self.fignum].remove_web_socket(self)
    if self.fignum == '0':
      # keep-alive died, so this whole session is over
      del app.prog_states[self.uid]
    else:
      # our figure is dead, delete it
      del app.fig_managers[self.fignum]

  def on_message(self, message):
    message = json.loads(message)
    if message['type'] == 'supports_binary':
      self.supports_binary = message['value']
    else:
      self.application.fig_managers[self.fignum].handle_json(message)

  def send_json(self, content):
    self.write_message(json.dumps(content))

  def send_binary(self, blob):
    if self.supports_binary:
      self.write_message(blob, binary=True)
    else:
      payload = blob.encode('base64').replace('\n', '')
      self.write_message("data:image/png;base64," + payload)


class _MockFigureManager(object):
  def __init__(self):
    self.web_sockets = set()

  def add_web_socket(self, ws):
    self.web_sockets.add(ws)

  def remove_web_socket(self, ws):
    self.web_sockets.remove(ws)
