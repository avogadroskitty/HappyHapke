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


phase_fn='legendre'
scatter_type='lambertian'
thetai=-30
thetae=0
n1=1.5725
Bg=0
small_file=''
medium_file=''
large_file=''
specwave_file=''
calspec_file=''
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
print('Initialization complete.', None, [fig])

#def preprocess(
low=0.32
high=2.55
UV=0.30
fit_order=1
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
print('Preprocessing complete: ', 'pp', [fig])


#def solve_for_k(
key='sml'
b=0.1
c=0.3
ff=0.00000000001
s=0
D=63
b, c, s, D, ff = map(float, (b, c, s, D, ff))
self2_guesses[key] = (b, c, s, D, ff)
traj = self2_pp_spectra[key]
plt.close('all')  # hack!
self2_ks[key] = analysis.MasterHapke1_PP(
    self2_hapke_scalar, traj, b, c, ff, s, D, debug_plots=True)
figures = [plt.figure(i) for i in plt.get_fignums()]
print('Solved for k: ', 'k-' + key, figures)
key='med'
b=0.2
c=0.4
ff=0.00000000001
s=0
D=106
b, c, s, D, ff = map(float, (b, c, s, D, ff))
self2_guesses[key] = (b, c, s, D, ff)
traj = self2_pp_spectra[key]
plt.close('all')  # hack!
self2_ks[key] = analysis.MasterHapke1_PP(
    self2_hapke_scalar, traj, b, c, ff, s, D, debug_plots=True)
figures = [plt.figure(i) for i in plt.get_fignums()]
print('Solved for k: ', 'k-' + key, figures)
key='big'
b=0.3
c=0.5
ff=0.00000000001
s=0
D=150
b, c, s, D, ff = map(float, (b, c, s, D, ff))
self2_guesses[key] = (b, c, s, D, ff)
traj = self2_pp_spectra[key]
plt.close('all')  # hack!
self2_ks[key] = analysis.MasterHapke1_PP(
    self2_hapke_scalar, traj, b, c, ff, s, D, debug_plots=True)
figures = [plt.figure(i) for i in plt.get_fignums()]
print('Solved for k: ', 'k-' + key, figures)

"""optimize_global_k"""
guess_key='med'
opt_strategy='slow'
lowb1=-1.71
lowb2=-1.72
lowb3=-1.73
upb1=1.71
upb2=1.72
upb3=1.73
lowc1=-1.01
lowc2=-1.02
lowc3=-1.03
upc1=1.01
upc2=1.02
upc3=1.03
lows1=0
lows2=0
lows3=0
ups1=0.061
ups2=0.062
ups3=0.063
lowD1=21
lowD2=36
lowD3=50
upD1=106
upD2=150
upD3=180
lowk=0
upk=0.1
num_solns=1
self2_hapke_vector_isow = self2_hapke_scalar.copy()
if self2_hapke_vector_isow.needs_isow:
  # use vector isow, instead of the scalar we had before
  _, high, UV = self2_pp_bounds
  idx1, idx2 = np.searchsorted(self2_calspec[:,0], (UV, high))
  self2_hapke_vector_isow.set_isow(self2_calspec[idx1:idx2,1])
 # set up initial guesses
k = self2_ks[guess_key]
guesses = np.empty(len(k) + 12)
ff = np.zeros(3)
for i, key in enumerate(('sml', 'med', 'big')):
  g = self2_guesses[key]
  guesses[i:12:3] = g[:4]
  ff[i] = g[4]
guesses[12:] = k
 # set up bounds
lb = np.empty_like(guesses)
lb[:12] = [lowb1, lowb2, lowb3, lowc1, lowc2, lowc3, lows1, lows2, lows3,
           lowD1, lowD2, lowD3]
lb[12:] = lowk
ub = np.empty_like(guesses)
ub[:12] = [upb1, upb2, upb3, upc1, upc2, upc3, ups1, ups2, ups3,
           upD1, upD2, upD3]
ub[12:] = upk
self2_bounds = (lb, ub)
 # solve
if opt_strategy == 'fast':
  solns = analysis.optimize_global_k(
      self2_hapke_vector_isow, self2_pp_spectra, guesses, lb, ub, ff,
      num_iters=int(num_solns))
  best_soln = solns[-1]
else:
  tmp = analysis.MasterHapke2_PP(
      self2_hapke_vector_isow, self2_pp_spectra, guesses, lb, ub, ff,
      tr_solver='lsmr', verbose=2, spts=int(num_solns))
  solns = [res.x for res in tmp]
  best_soln = min(tmp, key=lambda res: res.cost).x
 # save the best solution
self2_ks['global'] = best_soln[12:]
for i, key in enumerate(('sml', 'med', 'big')):
  b, c, s, D = best_soln[i:12:3]
  self2_guesses[key] = (b, c, s, D, ff[i])
  # plot solved parameters (b, c, s, D) for each grain size
fig1, axes = plt.subplots(figsize=(9,5), ncols=4, nrows=3, sharex=True,
                          frameon=False)
axes[0,0].set_ylabel('Small')
axes[1,0].set_ylabel('Medium')
axes[2,0].set_ylabel('Large')
axes[0,0].set_title('b')
axes[0,1].set_title('c')
axes[0,2].set_title('s')
axes[0,3].set_title('D')
for i, key in enumerate(('sml', 'med', 'big')):
  for j in range(4):
    ax = axes[i,j]
    idx = i + j*3
    ax.axhline(y=lb[idx], c='k', ls='dashed')
    ax.axhline(y=ub[idx], c='k', ls='dashed')
    vals = [guesses[idx]]
    vals.extend([sn[idx] for sn in solns])
    ax.plot(vals, marker='x')
for ax in axes[2]:
  ax.set_xlabel('Step #')
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
 # plot resulting rc vs original data for the best soln
fig2, (ax1, ax2) = plt.subplots(figsize=(9,4), ncols=2, sharex=True,
                                frameon=False)
best_soln = solns[-1]
line_colors = ['b', 'g', 'r']  # ['C0', 'C1', 'C3']
for i, key in enumerate(('sml', 'med', 'big')):
  wave, orig = self2_pp_spectra[key].T
  b, c, s, D = best_soln[i:12:3]
  scat = self2_hapke_vector_isow.scattering_efficiency(best_soln[12:], wave,
                                                      D, s)
  rc = self2_hapke_vector_isow.radiance_coeff(scat, b, c, ff[i])
  ax1.plot(wave, orig, color=line_colors[i], label=('%s grain' % key))
  ax1.plot(wave, rc, 'k--')
  ax1.set_xlabel('Wavelength (um)')
  ax1.set_ylabel('Reflectance (#)')
  ax1.set_title('Final fit')
  ax1.legend(fontsize='small', loc='best')
  ax2.plot(wave, np.abs(rc - orig), color=line_colors[i], lw=1,
           label=('%s fit' % key))
  ax2.set_title('Fit error')
  ax2.set_xlabel('Wavelength (um)')
  ax2.set_ylabel('Abs. Error')
  ax2.legend(fontsize='small', loc='best')
 # plot original ks vs global k
fig3, ax = plt.subplots(figsize=(6, 4), frameon=False)
ax.plot(wave, self2_ks['sml'], label='Small')
ax.plot(wave, self2_ks['med'], label='Medium')
ax.plot(wave, self2_ks['big'], label='Large')
ax.plot(wave, self2_ks['global'], 'k--', label='Global')
ax.set_xlabel('Wavelength (um)')
ax.set_title('Fitted k')
ax.legend(fontsize='small', loc='best')
msg = 'Finished %d iterations: ' % len(solns)
return msg, 'k-global', [fig1, fig2, fig3]