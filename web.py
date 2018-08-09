#!/usr/bin/env python
from __future__ import division, print_function
import io
import logging
import json
import os
import tornado.ioloop
import tornado.web
import tornado.websocket
from argparse import ArgumentParser
from matplotlib import rc
from matplotlib.backends.backend_webagg_core import (
    new_figure_manager_given_figure)
from socket import gethostname

from mplweb import MplWebApp
from prog_state import ProgramState

rc('figure', autolayout=True)
rc('lines', linewidth=1.5)
rc('axes.spines', top=False)
# rc('xtick', top=False)
# rc('ytick', right=False)

#Main Program that runs the server on the local machine on the specified port
def main():
  ap = ArgumentParser()
  ap.add_argument('--port', type=int, default=41414, help='Port. [%(default)s]')
  args = ap.parse_args()

  logging.basicConfig(level=logging.INFO)

  app = MplWebApp(
      [(r'/', HapkeHandler)],
      static_path=os.path.join(os.path.dirname(__file__), 'html'),
      debug=True)
  app.listen(args.port)
  print('Starting UI server at http://%s:%s/' % (gethostname(), args.port))
  try:
    tornado.ioloop.IOLoop.current().start()
  except KeyboardInterrupt:
    print('Server shutting down.')

#Handles the incoming request
class HapkeHandler(tornado.web.RequestHandler):
  def get(self):
    if bool(int(self.get_argument('dl', 0))):
      self._handle_download()
    else:
      self._init_program_state()

#Function to support download of the data files
  def _handle_download(self):
    uid = self.get_argument('uid')
    state = self.application.prog_states[uid]
    param = self.get_argument('p')
    fname, mimetype, data = state._download_data(param)
    self.set_header('Content-Type', mimetype)
    self.set_header('Content-Disposition', 'attachment; filename=' + fname)
    self.write(data)
    self.finish()

#Initial state of the program that loads the entire page from the ui.html file
  def _init_program_state(self):
    app = self.application
    # initialize the program state
    state = ProgramState()
    uid = format(id(state), 'x')
    app.prog_states[uid] = state
    # render
    self.render('ui.html', uid=uid, host=self.request.host)

#On clicking "Run" from the website - the post is called, based on the argument it calls the function at runtime
  def post(self):
    #Gets the userid for the user running the program
    uid = self.get_argument('uid')
    state = self.application.prog_states[uid]
    # collect arguments for this section - Changes based on section
    # Each section has a hidden variable in ui.html that holdds the value to be passed in the section variable
    # When the submit button is clicked - the input tags in html are sent to the server
    # These values can be received in python using the below line.
    section = self.get_argument('section')
    kwargs = self._collect_kwargs(ignored_keys=('uid', 'section'))
    # run the section method
    logging.info('Running %s: %r', section, kwargs)
    try:
      #Calling each function at run time
      message, dl_param, figures = getattr(state, section)(**kwargs)
    except EnvironmentError as e:
      logging.exception('Section %s failed.', section)
      self.set_status(400)
      self.write(e.strerror)
      return
    except Exception as e:
      logging.exception('Section %s failed.', section)
      self.set_status(400)
      self.write(e.message)
      return
    # start returning html to the frontend
    self.write('<div>')
    self.write('<input type="hidden" id="uid_val" value="%s" />' % uid);
    if message:
      self.write(message)
    if dl_param:
      self.write('<a href="/?dl=1&uid=%s&p=%s" target="_blank">Download</a>' %
                 (uid, dl_param))
    self.write('</div>')
    # initialize the figure managers
    fig_managers = self.application.fig_managers
    for fig in figures:
      # take care to prevent a fignum of zero, which is special to us
      fignum = id(fig) * 10 + 1
      fig_managers[str(fignum)] = new_figure_manager_given_figure(fignum, fig)
      self.write('<div id="fig%s" class="figure"></div>' % fignum)

  def _collect_kwargs(self, ignored_keys=()):
    kwargs = {}
    # look at all the request parameters
    for key in set(self.request.arguments) - set(ignored_keys):
      if key.endswith('[]'):
        kwargs[key[:-2]] = self.get_arguments(key)
      else:
        kwargs[key] = self.get_argument(key)
    # file arguments are treated specially
    for key, files in self.request.files.items():
      if key.endswith('[]'):
        key = key[:-2]
      filedata = [io.BytesIO(files[0]['body']) for f in files]
      if len(files) == 1:
        kwargs[key] = filedata[0]
      else:
        kwargs[key] = filedata
    return kwargs

if __name__ == '__main__':
  main()
