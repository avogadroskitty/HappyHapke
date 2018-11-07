# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:27:47 2018

@author: ecskl
"""

from __future__ import division, print_function
import numpy as np
import os
from io import BytesIO
from glob import glob
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from zipfile import ZipFile

import analysis
from hapke_model import get_hapke_model


def initialize(phase_fn='legendre', scatter_type='lambertian',
               thetai=0, thetae=0, n1=0, Bg=0,
               small_file='', medium_file='', large_file='',
               specwave_file='', calspec_file=''):
  # HACK: use defaults if some/all files aren't provided
  specwave_file = specwave_file or '../data/specwave2.mat'
  calspec_file = calspec_file or '../data/calspecw2.mat'
  # small_file = small_file or '../data/kjs.mat'
  # medium_file = medium_file or '../data/kjm.mat'
  # large_file = large_file or '../data/kjb.mat'
  small_file = small_file or '../data/BytWS63106i30e0.asc'
  medium_file = medium_file or '../data/BytWM106150i30e0.asc'
  large_file = large_file or '../data/BytWB150180i30e0.asc'
   # initialize the model
  HapkeModel = get_hapke_model(phase_fn=phase_fn, scatter=scatter_type)
  thetai, thetae = np.deg2rad([float(thetai), float(thetae)])
  self2_hapke_scalar = HapkeModel(thetai, thetae, float(n1), float(Bg))

  self2_spectra = {}
  for key, infile in [('sml', small_file),
                      ('med', medium_file),
                      ('big', large_file)]:
    self2_spectra[key] = analysis.loadmat_single(infile)
   # data to be filled in (later) for each grain size
  self2_pp_spectra = {}
  self2_ks = {}
  self2_guesses = {}

  if self2_hapke_scalar.needs_isow:
      # store the calibration spectrum
    specwave = analysis.loadmat_single(specwave_file).ravel()
    calspec = analysis.loadmat_single(calspec_file).ravel()
    self2_calspec = np.column_stack((specwave, calspec))

    # plot the loaded spectra
  num_plots = 2 if self2_hapke_scalar.needs_isow else 1
  fig = Figure(figsize=(9, 4), frameon=False, tight_layout=True)
  ax1 = fig.add_subplot(1, num_plots, 1)
  ax1.plot(*self2_spectra['sml'].T, label='Small grain')
  ax1.plot(*self2_spectra['med'].T, label='Medium grain')
  ax1.plot(*self2_spectra['big'].T, label='Large grain')
  ax1.set_xlabel('Wavelength ($\mu{}m)$')
  ax1.set_ylabel('Reflectance')
  ax1.legend(fontsize='small', loc='best')
  ax1.set_title('Input VNIR spectra')
  if self2_hapke_scalar.needs_isow:
    ax2 = fig.add_subplot(1, num_plots, 2)
    ax2.plot(specwave, calspec, 'k-')
    ax2.set_title('Calibrated standard')
    ax2.set_xlabel('Wavelength ($\mu{}m)$')
  # return html for a plot of the inputs + calibration
  return 'Initialization complete.', None, [fig]

def preprocess(low=0, high=0, UV=0, fit_order=0):
  low, high, UV = float(low), float(high), float(UV)
  self2_pp_bounds = (low, high, UV)
  fit_order = int(fit_order)

  if self2_hapke_scalar.needs_isow:
    # initialize isow as a scalar
    isoind1, isoind2 = np.searchsorted(self2_calspec[:,0], (low, high))
    self2_hapke_scalar.set_isow(self2_calspec[isoind1:isoind2,1].mean())
   # run preprocessing on each spectrum
  for key, traj in self2_spectra.items():
    self2_pp_spectra[key] = analysis.preprocess_traj(traj, low, high, UV,
                                                    fit_order=fit_order)
   # plot the results
  fig = Figure(figsize=(6, 4), frameon=False, tight_layout=True)
  ax = fig.gca()
  ax.plot(*self2_pp_spectra['sml'].T, label='Small grain')
  ax.plot(*self2_pp_spectra['med'].T, label='Medium grain')
  ax.plot(*self2_pp_spectra['big'].T, label='Large grain')
  ax.legend(fontsize='small', loc='best')
  ax.set_title('Preprocessed spectra')
  ax.set_xlabel('Wavelength ($\mu{}m)$')
  ax.set_ylabel('Reflectance')
  return 'Preprocessing complete: ', 'pp', [fig]

def solve_for_k(key='sml', b=0, c=0, ff=0, s=0, D=0):
  b, c, s, D, ff = map(float, (b, c, s, D, ff))
  self2_guesses[key] = (b, c, s, D, ff)
  traj = self2_pp_spectra[key]
  plt.close('all')  # hack!
  self2_ks[key] = analysis.MasterHapke1_PP(
      self2_hapke_scalar, traj, b, c, ff, s, D, debug_plots=True)
  figures = [plt.figure(i) for i in plt.get_fignums()]
  return 'Solved for k: ', 'k-' + key, figures