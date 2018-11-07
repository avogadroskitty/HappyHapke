# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:27:59 2018

@author: ecskl
"""

#this is what I am doing
#Prog_state

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


"""INITIALIZE FUNCTION"""

phase_fn='legendre'
scatter_type='lambertian'
thetai=-30
thetae=0
n1=1.5725
Bg=0
#small_file=''
#medium_file=''
#large_file=''
specwave_file=''
calspec_file=''
file1=''
file2=''
file3=''
file4=''
file5=''
file6=''
file7=''
file8=''
file9=''
file10=''
  
  # HACK: use defaults if some/all files aren't provided
specwave_file = specwave_file or '../data/specwave2.mat'
calspec_file = calspec_file or '../data/calspecw2.mat'
  # small_file = small_file or '../data/kjs.mat'
  # medium_file = medium_file or '../data/kjm.mat'
  # large_file = large_file or '../data/kjb.mat'

file1 = file1 or '../data/BytWS63106i30e0.asc' 
file2 = file2 or '../data/BytWM106150i30e0.asc'
file3 = file3 or '../data/BytWB150180i30e0.asc'

    #-^Above^-Default file names defined here also used in ui.html for solve for K - dropdown. If changed here must be changed there as well.

self_file_key_list = ['file4','file5','file6','file7','file8','file9','file10']
    # initialize the model
HapkeModel = get_hapke_model(phase_fn=phase_fn, scatter=scatter_type)
thetai, thetae = np.deg2rad([float(thetai), float(thetae)])
self_hapke_scalar = HapkeModel(thetai, thetae, float(n1), float(Bg))
  
self_spectra = {}
for key, infile in [('file1', file1),
                    ('file2', file2),
                    ('file3', file3),
                    ('file4', file4),
                    ('file5', file5),
                    ('file6', file6),
                    ('file7', file7),
                    ('file8', file8),
                    ('file9', file9),
                    ('file10', file10)]:
      #Checks if the infile variable has a string -- sanity check if not all ten files are uploaded
      #self.spectra has the number of grain files included in the process
   if not infile == '':
     self_spectra[key] = analysis.loadmat_single(infile)

    # data to be filled in (later) for each grain size
self_pp_spectra = {}
self_ks = {}
self_guesses = {}
self_scat_eff_grain = {}

if self_hapke_scalar.needs_isow:
      # store the calibration spectrum
  specwave = analysis.loadmat_single(specwave_file).ravel()
  calspec = analysis.loadmat_single(calspec_file).ravel()
  self_calspec = np.column_stack((specwave, calspec))

    # plot the loaded spectra
num_plots = 2 if self_hapke_scalar.needs_isow else 1
fig = Figure(figsize=(9, 4), frameon=False, tight_layout=True)
ax1 = fig.add_subplot(1, num_plots, 1)

    #Three default files loaded into the spectra dictionary are plotted on the graph
ax1.plot(*self_spectra['file1'].T, label='Small grain')
ax1.plot(*self_spectra['file2'].T, label='Medium grain')
ax1.plot(*self_spectra['file3'].T, label='Large grain')
    
    #Adding plots for files uploaded - can upload maximum of 10 files including the three default
for k in self_file_key_list:
  if k in self_spectra:
    ax1.plot(*self_spectra[k].T, label=k)

ax1.set_xlabel('Wavelength ($\mu{}m)$')
ax1.set_ylabel('Reflectance')
ax1.legend(fontsize='small', loc='best')
ax1.set_title('Input VNIR spectra')
if self_hapke_scalar.needs_isow:
  ax2 = fig.add_subplot(1, num_plots, 2)
  ax2.plot(specwave, calspec, 'k-')
  ax2.set_title('Calibrated standard')
  ax2.set_xlabel('Wavelength ($\mu{}m)$')
  
  
  
"""PREPROCESSING FUNCTION"""
low=0.32
high=2.55
UV=0.32
fit_order=1
low, high, UV = float(low), float(high), float(UV)
self_pp_bounds = (low, high, UV)
fit_order = int(fit_order)

if self_hapke_scalar.needs_isow:
      # initialize isow as a scalar
  isoind1, isoind2 = np.searchsorted(self_calspec[:,0], (low, high))
  self_hapke_scalar.set_isow(self_calspec[isoind1:isoind2,1].mean())

    # run preprocessing on each spectrum
    #self_Spectra.items() is both columns of all three grains printing 3, 2, 1
    #traj is just the columns
    #for instance, 
    #for key, traj in self_spectra.items():
    #  print(key)
    #  print(traj)
    #will print file3, then xy data etc....
for key, traj in self_spectra.items():
  self_pp_spectra[key] = analysis.preprocess_traj(traj, low, high, UV,
                                                      fit_order=fit_order)

    # plot the results
fig = Figure(figsize=(6, 4), frameon=False, tight_layout=True)
ax = fig.gca()
ax.plot(*self_pp_spectra['file1'].T, label='Small grain')
ax.plot(*self_pp_spectra['file2'].T, label='Medium grain')
ax.plot(*self_pp_spectra['file3'].T, label='Large grain')

    #If additional files exist we plot them 
for k in self_file_key_list:
  if k in self_pp_spectra:
    ax.plot(*self_pp_spectra[k].T, label=k)

ax.legend(fontsize='small', loc='best')
ax.set_title('Preprocessed spectra')
ax.set_xlabel('Wavelength ($\mu{}m)$')
ax.set_ylabel('Reflectance')
   #pp is the parameter used for identifying the download data.
print('Preprocessing complete: ', 'pp', [fig])



"""SOLVE_FOR_K FUNCTION"""
  #Section Two by Default, Section Three and Four - Matlab Code
key='file1'
b=0.1
c=0.3
ff=0.000000001
s=0
D=63
b, c, s, D, ff = map(float, (b, c, s, D, ff))
self_guesses[key] = (b, c, s, D, ff)
traj = self_pp_spectra[key]
plt.close('all')  # hack!
    #The hidden treasure where all the brains are hidden
solved_k, scat_eff = analysis.MasterHapke1_PP(
    self_hapke_scalar, traj, b, c, ff, s, D, debug_plots=True)

self_ks[key] = solved_k
self_scat_eff_grain[key] = scat_eff
figures = [plt.figure(i) for i in plt.get_fignums()]
print('Solved for k: ', 'sk-' + key, figures)
"""SOLVE_FOR_K FUNCTION"""
  #Section Two by Default, Section Three and Four - Matlab Code
key='file2'
b=0.2
c=0.4
ff=0.000000001
s=0
D=106
b, c, s, D, ff = map(float, (b, c, s, D, ff))
self_guesses[key] = (b, c, s, D, ff)
traj = self_pp_spectra[key]
plt.close('all')  # hack!
    #The hidden treasure where all the brains are hidden
solved_k, scat_eff = analysis.MasterHapke1_PP(
    self_hapke_scalar, traj, b, c, ff, s, D, debug_plots=True)

self_ks[key] = solved_k
self_scat_eff_grain[key] = scat_eff
figures = [plt.figure(i) for i in plt.get_fignums()]
print('Solved for k: ', 'sk-' + key, figures)
"""SOLVE_FOR_K FUNCTION"""
  #Section Two by Default, Section Three and Four - Matlab Code
key='file3'
b=0.3
c=0.5
ff=0.000000001
s=0
D=150
b, c, s, D, ff = map(float, (b, c, s, D, ff))
self_guesses[key] = (b, c, s, D, ff)
traj = self_pp_spectra[key]
plt.close('all')  # hack!
    #The hidden treasure where all the brains are hidden
solved_k, scat_eff = analysis.MasterHapke1_PP(
    self_hapke_scalar, traj, b, c, ff, s, D, debug_plots=True)

self_ks[key] = solved_k
self_scat_eff_grain[key] = scat_eff
figures = [plt.figure(i) for i in plt.get_fignums()]
print('Solved for k: ', 'sk-' + key, figures)

"""Optimize_Global_k FUNCTION"""
guess_key='file2'
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

    #The previous step only approximates for a single grain size
    #Should we have guesses for all grain samples or only the ones we have approximated for?
no_of_grain_samples = len(self_spectra)
total_guesses = no_of_grain_samples * 4 # 4 values (b,c,s,D) for each grain size

self_hapke_vector_isow = self_hapke_scalar.copy()
if self_hapke_vector_isow.needs_isow:
      # use vector isow, instead of the scalar we had before
  _, high, UV = self_pp_bounds
  idx1, idx2 = np.searchsorted(self_calspec[:,0], (UV, high))
  self_hapke_vector_isow.set_isow(self_calspec[idx1:idx2,1])

    # set up initial guesses
k = self_ks[guess_key]
  # [215,] -- size of the array
guesses = np.empty(len(k) + total_guesses)
  #[215 + (4 * no of grains),] - size of the guesses list
ff = np.zeros(no_of_grain_samples)
for i, key in enumerate(sorted(self_guesses.keys())):
  g = self_guesses[key]
      # Unpacks the b,c,s,D values for each grain size into one large array. g holds b,c,s,D,f -- we take only the first four
  guesses[i:total_guesses:no_of_grain_samples] = g[:4] 
      # guesses Example:
      # for sml, med anf big grain sizes
      # [sml-b, med-b, big-g, sml-C, med-C, big-C, sml-S, med-S, big-S, sml-D, med-D, big-D, 215 values of K]
      # total with the length of K - 4 values for each grain size -- this is the magic 12
  ff[i] = g[4]
guesses[total_guesses:] = k #Filling the rest of the array with the value of K
print(self_ks)
    # set up bounds
lb = np.empty_like(guesses)
    #Values that will be there regardless if additional grain sizes are uploaded
lb[:12] = [lowb1, lowb2, lowb3, lowc1, lowc2, lowc3, lows1, lows2, lows3,
lowD1, lowD2, lowD3]

temp_low_bound = []

#this needs to change for new number scheme - 'grain' won't work
for grain in self_spectra.keys():
  if grain not in ['file1', 'file2', 'file3']:
    temp_low_bound.append(kwargs['lowb'+grain])
    temp_low_bound.append(kwargs['lowc'+grain])
    temp_low_bound.append(kwargs['lows'+grain])
    temp_low_bound.append(kwargs['lowD'+grain])

lb[12:total_guesses] = temp_low_bound
    #Filling in rest of the values
lb[total_guesses:] = lowk

ub = np.empty_like(guesses)
ub[:12] = [upb1, upb2, upb3, upc1, upc2, upc3, ups1, ups2, ups3,
upD1, upD2, upD3]
ub[12:] = upk
# does self_bounds ever get used?
self_bounds = (lb, ub)

    # solve
tmp = analysis.MasterHapke2_PP(
    self_hapke_vector_isow, self_pp_spectra, guesses, lb, ub, ff,
    tr_solver='lsmr', verbose=2, spts=int(num_solns))
solns = [res.x for res in tmp]
best_soln = min(tmp, key=lambda res: res.cost).x

    # save the best solution
self_ks['global'] = best_soln[total_guesses:]
for i, key in enumerate(sorted(self_spectra.keys())):
  b, c, s, D = best_soln[i:total_guesses:no_of_grain_samples]
  self_guesses[key] = (b, c, s, D, ff[i])

    # plot solved parameters (b, c, s, D) for each grain size
fig1, axes = plt.subplots(figsize=(9,5), ncols=4, nrows=no_of_grain_samples, sharex=True,
                              frameon=False)

    #Label the rows
for i, key in self_spectra.keys():
  axes[i,0].set_ylabel(key)
    
    #Label the columns
axes[0,0].set_title('b')
axes[0,1].set_title('c')
axes[0,2].set_title('s')
axes[0,3].set_title('D')
for i, key in enumerate(sorted(self_spectra.keys())):
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
for i, key in enumerate(sorted(self_spectra.keys())):
  wave, orig = self_pp_spectra[key].T
  b, c, s, D = best_soln[i:total_guesses:no_of_grain_samples]
  scat = self_hapke_vector_isow.scattering_efficiency(best_soln[total_guesses:], wave,
                                                          D, s)
  rc = self_hapke_vector_isow.radiance_coeff(scat, b, c, ff[i])

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
for key in self_spectra.keys():
  ax.plot(wave, self_ks[key], label=key)

ax.plot(wave, self_ks['global'], 'k--', label='Global')
ax.set_xlabel('Wavelength (um)')
ax.set_title('Fitted k')
ax.legend(fontsize='small', loc='best')

msg = 'Finished %d iterations: ' % len(solns)
print(msg, 'k-global', [fig1, fig2, fig3])