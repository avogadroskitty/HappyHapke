#!/usr/bin/env python
from __future__ import division, print_function
import numpy as np
import os
import random
import scipy.io
import yaml
from argparse import ArgumentParser
from scipy.integrate import trapz
from matplotlib import pyplot as plt

from hapke import get_hapke_model


def main():
  ap = ArgumentParser()
  ap.add_argument('config', default='config-byt.yml', nargs='?',
                  help='Path to YAML config file. [%(default)s]')
  args = ap.parse_args()

  model, config, wave = setup(args.config)
  fig, (ax1, ax2) = plt.subplots(ncols=2)
  ax1.set_title('k')
  ax1.set_xlabel('wavelength (microns)')
  ax2.set_title('n')
  ax2.set_xlabel('frequency (wavenumber)')

  # find a good starting point for the dispersion vector (k)
  k = _guess_dispersion(model, config, wave)
  ax1.plot(wave, k, label='guessed')

  # solve for a grainsize-independent dispersion vector
  k = _optimize_dispersion(model, config, wave, k)
  ax1.plot(wave, k, label='optimized')

  # concatenate MIR dispersion data
  wave, k = _combine_dispersion(config, wave, k)
  ax1.plot(wave, k, label='combined w/ MIR')

  # resample k in frequency space
  freq, wave, k = _convert_to_frequency(wave, k)

  # compute full index of refraction
  n = _kramers_kronig(config, freq, k)
  ax2.plot(freq, n, label='kramers kronig')

  # TODO: solve for parameterized dispersion with full refraction
  # Do we convert back to wavelengths? yes
  # Should we downsample? no
  # allow k to scale instead of full vector?
  # use single (b,c) for all data, vary over wavelength
  ax1.legend()
  ax2.legend()
  plt.show()


def setup(config_file):
  config = yaml.safe_load(open(config_file))
  data_dir = os.path.join(os.path.dirname(__file__), config['data dir'])
  if not (os.path.exists(data_dir) and os.path.isdir(data_dir)):
    raise ValueError('Invalid data directory: ' + data_dir)

  # load and preprocess VNIR spectra
  pp_cfg = config['preprocessing']
  for s in config['spectra']:
    path = os.path.join(data_dir, s['filename'])
    wave, reflect = _prep_spectrum(path, pp_cfg, extrapolate=True)
    s['reflect'] = reflect
    # convert angles to radians
    s['thetai'] = np.deg2rad(s['thetai'])
    s['thetae'] = np.deg2rad(s['thetae'])

  # load MIR dispersion data
  disp_cfg = config['dispersion']
  mir_k = loadmat_single(os.path.join(data_dir, disp_cfg['MIR k'])).ravel()
  mir_freq = loadmat_single(os.path.join(data_dir, disp_cfg['MIR v'])).ravel()
  disp_cfg['MIR wavelengths'] = 10000 / mir_freq
  disp_cfg['MIR dispersion'] = mir_k

  # set up the simplified model with a scalar refraction index
  phase_cfg = config['phase function']
  scat_cfg = config['scattering model']
  HapkeModel = get_hapke_model(phase_fn=phase_cfg['type'],
                               scatter=scat_cfg['type'])
  params_cfg = config['parameters']
  model = HapkeModel(params_cfg['mean reflectance'],
                     params_cfg['opposition surge'])

  if model.needs_isow:
    path = os.path.join(data_dir, scat_cfg['calibration spectrum'])
    wave, isow = _prep_spectrum(path, pp_cfg, extrapolate=False)
    good_region = slice(*np.searchsorted(wave, (0.5, 1.25)))
    scat_cfg['isow'] = isow
    scat_cfg['mean isow'] = isow[good_region].mean()

  return model, config, wave


def _guess_dispersion(model, config, wave):
  phase_cfg = config['phase function']
  scat_cfg = config['scattering model']

  # TODO: choose a spectrum in the center of the parameter space
  spectrum = random.choice(config['spectra'])
  reflect = spectrum['reflect']
  thetai = spectrum['thetai']
  thetae = spectrum['thetae']

  # create table of increasing single scattering albedo and use linear
  # interpolation to solve backwards from the real reflectance data
  table_size = len(wave) * 2
  phase_params = [c['guess'] for c in phase_cfg['coefficients']]
  Pg = model.single_particle_phase(thetai, thetae, *phase_params)
  if scat_cfg['type'] == 'isotropic':
    scatter_params = [scat_cfg['filling factor'], scat_cfg['mean isow']]
  else:
    scatter_params = []
  scat_table = np.linspace(0, 1, table_size, endpoint=False)
  table_rc = model.radiance(thetai, thetae, scat_table, Pg, *scatter_params)
  scat_eff = np.interp(reflect, table_rc, scat_table)

  # use the same trick to back-solve for k from scat_eff,
  # except we plug in the (k/wavelength) term when interpolating
  # TODO: incorporate bounds on k to determine table bounds
  s = scat_cfg['internal scattering']['guess']
  D = spectrum['grain size']['guess']
  k_wave = np.logspace(-1, -7, table_size)
  table_scat = model.scattering_efficiency(1, k_wave, D, s)
  return np.interp(scat_eff, table_scat, k_wave) * wave


def _optimize_dispersion(model, config, wave, k):
  # guesses = np.concatenate((args.b, args.c, args.s, args.D, k))
  # lb = np.concatenate((args.lowb, args.lowc, args.lows, args.lowD,
  #                      np.zeros_like(k) + args.lowk))
  # ub = np.concatenate((args.upb, args.upc, args.ups, args.upD,
  #                      np.zeros_like(k) + args.upk))
  # solutions = analysis.MasterHapke2_PP(hapke, spectra, guesses, lb, ub,
  #                                      args.ff, tr_solver='lsmr', verbose=2)
  return k


def _combine_dispersion(config, vnir_wave, vnir_k):
  disp_cfg = config['dispersion']
  mir_wave = disp_cfg['MIR wavelengths']
  mir_k = disp_cfg['MIR dispersion']

  # ensure MIR data is in ascending wavelength order
  if mir_wave[0] > mir_wave[-1]:
    mir_wave = mir_wave[::-1]
    mir_k = mir_k[::-1]

  # extend vnir data to meet mir data, if needed
  vnir_wave, vnir_k = _extrapolate(vnir_wave, vnir_k,
                                   (vnir_wave[0], mir_wave[0]),
                                   fit_order=3, fit_length=50)

  # correct for offset in MIR data
  mir_k += (vnir_k[-1] - mir_k[0])

  # remove any negative data (take up to first negative value)
  neg_idx, = np.where(mir_k < 0)
  if len(neg_idx) > 0:
    mir_wave = mir_wave[:neg_idx[0]]
    mir_k = mir_k[:neg_idx[0]]

  # combine
  wave = np.concatenate((vnir_wave, mir_wave))
  k = np.concatenate((vnir_k, mir_k))
  return wave, k


def _convert_to_frequency(wave, k):
  # wavelength -> frequency
  freq = 10000 / wave
  # resample frequencies to fixed step-size
  v = np.linspace(freq[0], freq[-1], len(freq))
  k = np.interp(v, freq, k)
  wave = 10000 / v
  return v, wave, k


def _kramers_kronig(config, v, k, num_intervals=100):
  """This section determines the real index of refraction (n) from k,
  using a singly subtractive Kramers Kronig calculation."""
  params_cfg = config['parameters']
  v1 = 10000 / params_cfg['anchor wavelength']

  # make start point and end point values for integration that place k at the
  # center of each range
  dv = v[1] - v[0]
  dx = dv / num_intervals
  halfv = dv / 2
  offset = np.linspace(-halfv, halfv, num_intervals)
  xx = v + offset[:,None]

  # compute all the bits that don't change in the loop
  v_sq = v**2
  v1_sq = v1**2
  n = (2/np.pi) * (v_sq - v1_sq) * dx
  xx_sq = xx ** 2
  numerator = xx * k / (xx_sq - v1_sq)

  # TODO: look into vectorizing this fully. It's not too hard,
  # but might be more memory than we'd like to spend.
  for i, v_sq_i in enumerate(v_sq):
    # Compute function over all xx for each v_j
    yy = numerator / (xx_sq - v_sq_i)
    # calculate the real index of refraction with integral over the grid
    n[i] *= trapz(yy, axis=0).sum()

  n += params_cfg['mean reflectance']
  return n


def _prep_spectrum(fpath, pp_cfg, extrapolate=True):
  w, x = loadmat_single(fpath).T

  # make sure wavelength is in microns
  if w[0] > 100:
      w /= 1000

  extend_cfg = pp_cfg['extrapolate']
  if extrapolate:
    w, x = _crop_spectrum(w, x, pp_cfg['crop'])
    w, x = _extrapolate(w, x, extend_cfg['bounds'],
                        fit_order=extend_cfg['polynomial order'])
  else:
    lb = min(pp_cfg['crop'][0], extend_cfg['bounds'][0])
    ub = max(pp_cfg['crop'][1], extend_cfg['bounds'][1])
    w, x = _crop_spectrum(w, x, (lb, ub))
  return w, x


def loadmat_single(filename):
  """Loads one array from a MAT-file."""
  if filename.endswith('.asc'):
    return np.loadtxt(filename)
  mat = scipy.io.loadmat(filename, appendmat=False)
  keys = [k for k in mat.keys() if not k.startswith('_')]
  if len(keys) != 1:
    raise ValueError('loadmat_single expects one variable, got %r' % keys)
  return mat[keys[0]]


def _crop_spectrum(w, x, bounds):
  idx1, idx2 = np.searchsorted(w, bounds)
  region = slice(idx1, idx2 + 1)
  return w[region], x[region]


def _extrapolate(w, x, bounds, fit_order=0, fit_length=100):
  dw = w[1] - w[0]
  left_w = np.arange(bounds[0], w[0], dw)
  right_w = np.arange(w[-1], bounds[1], dw)
  if len(left_w) > 0 and fit_order > 0:
    # TODO: make this depend on dw
    poly = np.poly1d(np.polyfit(w[:fit_length], x[:fit_length], fit_order))
    left_x = poly(left_w)
  else:
    # pad with whatever the end value is
    left_x = np.full_like(left_w, x[0], dtype=x.dtype)
  if len(right_w) > 0 and fit_order > 0:
    # TODO: make this depend on dw
    poly = np.poly1d(np.polyfit(w[-fit_length:], x[-fit_length:], fit_order))
    right_x = poly(right_w)
  else:
    # pad with whatever the end value is
    right_x = np.full_like(right_w, x[-1], dtype=x.dtype)
  comb_w = np.concatenate((left_w, w, right_w))
  comb_x = np.concatenate((left_x, x, right_x))
  return comb_w, comb_x


if __name__ == '__main__':
  main()
