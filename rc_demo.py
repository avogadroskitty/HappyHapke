#!/usr/bin/env python
from __future__ import division, print_function
import numpy as np
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.template
from argparse import ArgumentParser
from matplotlib.backends.backend_webagg_core import (
    new_figure_manager_given_figure as make_fig_manager)
from matplotlib.figure import Figure
from socket import gethostname

import analysis
from hapke_model import get_hapke_model
from mplweb import MplWebApp

demo_tpl = tornado.template.Template('''
<html><head>
<title>Hapke RC Demo</title>
<link rel="stylesheet" href="_static/css/page.css" type="text/css">
<link rel="stylesheet" href="_static/css/boilerplate.css" type="text/css">
<link rel="stylesheet" href="_static/css/fbm.css" type="text/css">
<link rel="stylesheet" href="_static/jquery/css/themes/base/jquery-ui.min.css">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js">
</script>
<script
src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.11.4/jquery-ui.min.js">
</script>
<script src="mpl.js"></script>
<script type="text/javascript">
function savefig(figure, format) {
  window.open('/'+figure.id+'/download.'+format, '_blank');
}
var keepalive, fig, websocket_type = mpl.get_websocket_type();
$(document).ready(function() {
  // set up figure
  var ws = new websocket_type('ws://{{host}}/{{uid}}/{{figid}}/ws');
  fig = new mpl.figure('{{figid}}', ws, savefig, $('#fig'));
  // open keep-alive websocket
  keepalive = new websocket_type('ws://{{host}}/{{uid}}/0/ws');
  keepalive.onclose = function(){
    // close the figure by replacing its canvas with a static image
    var img = new Image();
    img.src = $('#fig').find('.mpl-canvas')[0].toDataURL('image/png');
    $('#fig').html(img);
  };
  // catch slider changes
  var form = $('form')[0];
  $('input').change(function(){
    $.ajax({
      url: '/',
      data: new FormData(form),
      processData: false,
      contentType: false,
      type: 'POST'
    });
  }).change();
});
</script>
<style type="text/css">
form { padding: 1em; }
p { font-weight: bold; }
label { display: block; }
output {
  border: 1px solid black;
  width: 3em;
  display: inline-block;
  text-align: right;
  padding-right: 0.25em;
  background-color: lightsteelblue;
}
</style>
</head><body>
<div id="fig"></div>
<form method='POST'>
<p>Hapke parameters</p>
<label>
  <output id='b_out'>0.10</output>
  <input name='b' type='range' min='-1.7' max='1.7' value='0.1' step='0.05'
   oninput='b_out.value = (+this.value).toFixed(2)' />
  Coefficient b of Legendre polynomial.
</label>
<label>
  <output id='c_out'>0.3</output>
  <input name='c' type='range' min='-1' max='1' value='0.3' step='0.1'
   oninput='c_out.value = (+this.value).toFixed(1)' />
  Coefficient c of Legendre polynomial.
</label>
<label>
  <output id='ff_out'>0.46</output>
  <input name='ff' type='range' min='0.01' max='0.7' value='0.46' step='0.01'
   oninput='ff_out.value = (+this.value).toFixed(2)' />
  Filling factor.
</label>
<label>
  <output id='s_out'>0.000</output>
  <input name='s' type='range' min='0' max='0.06' value='0' step='0.005'
   oninput='s_out.value = (+this.value).toFixed(3)' />
  Internal scattering.
</label>
<label>
  <output id='D_out'>63</output>
  <input name='D' type='range' min='10' max='125' value='63' min='0'
   oninput='D_out.value = this.value' />
  Grain size.
</label>
</form></body></html>
''')

MODEL = get_hapke_model()(np.deg2rad(-30), 0, 1.76575, 0)
SPECTRA = {}
K = {}
RC_LINE = []


def main():
  ap = ArgumentParser()
  ap.add_argument('--port', type=int, default=8888, help='Port. [%(default)s]')
  args = ap.parse_args()

  # initialize isow as the mean of a fixed range
  specwave = analysis.loadmat_single('../data/specwave2.mat').ravel()
  calspec = analysis.loadmat_single('../data/calspecw2.mat').ravel()
  isoind1, isoind2 = np.searchsorted(specwave, (0.5, 1.25))
  MODEL.set_isow(calspec[isoind1:isoind2].mean())

  # load spectrum data
  SPECTRA['file1'] = analysis.loadmat_single('../data/kjs.mat')
  SPECTRA['file2'] = analysis.loadmat_single('../data/kjm.mat')
  SPECTRA['file3'] = analysis.loadmat_single('../data/kjb.mat')
  for k in SPECTRA.keys():
    SPECTRA[k] = analysis.preprocess_traj(SPECTRA[k], 0.405, 2.451, 0.301)

  K['file2'] = analysis.MasterHapke1_PP(MODEL, SPECTRA['file2'], .1, .3, .46, 0, 63)

  app = MplWebApp([(r'/', DemoHandler)], debug=True)
  app.listen(args.port)
  print('Starting UI server at http://%s:%s/' % (gethostname(), args.port))
  tornado.ioloop.IOLoop.current().start()


class DemoHandler(tornado.web.RequestHandler):
  def get(self):
    uid, figid = 'deadbeef', '1'
    self.application.prog_states[uid] = None
    fig = Figure(frameon=False, tight_layout=True)
    ax = fig.gca()
    for k, traj in SPECTRA.iteritems():
      ax.plot(*traj.T, label=k)
    ax.legend()

    RC_LINE.extend(ax.plot(traj[:,0], traj[:,1]*0, 'k--'))
    self.application.fig_managers[figid] = make_fig_manager(figid, fig)
    self.write(demo_tpl.generate(uid=uid, figid=figid, host=self.request.host))

  def post(self):
    b = float(self.get_argument('b'))
    c = float(self.get_argument('c'))
    s = float(self.get_argument('s'))
    D = float(self.get_argument('D'))
    ff = float(self.get_argument('ff'))

    wave = SPECTRA['file2'][:,0]
    k = K['file2']

    scat = MODEL.scattering_efficiency(k, wave, D, s)
    rc = MODEL.radiance_coeff(scat, b, c, ff)

    canvas = self.application.fig_managers['1'].canvas
    RC_LINE[0].set_ydata(rc)
    canvas.draw()


if __name__ == '__main__':
  main()
