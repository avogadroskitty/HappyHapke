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


class ProgramState(object):
  def initialize(self, phase_fn='legendre', scatter_type='isotropic',
                 thetai=0, thetae=0, n1=0, Bg=0,
                 small_file='', medium_file='', large_file='',
                 specwave_file='', calspec_file='', 
                 file1='', file2='', file3='', file4='', file5='', file6='', file7=''):  
    # HACK: use defaults if some/all files aren't provided
    specwave_file = specwave_file or '../data/specwave2.mat'
    calspec_file = calspec_file or '../data/calspecw2.mat'
    # small_file = small_file or '../data/kjs.mat'
    # medium_file = medium_file or '../data/kjm.mat'
    # large_file = large_file or '../data/kjb.mat'

    small_file = small_file or '../data/BytWS63106i30e0.asc' 
    medium_file = medium_file or '../data/BytWM106150i30e0.asc'
    large_file = large_file or '../data/BytWB150180i30e0.asc'

    #-^Above^-Default file names defined here also used in ui.html for solve for K - dropdown. If changed here must be changed there as well.

    self.file_key_list = ['file1','file2','file3','file4','file5','file6','file7']
    # initialize the model
    HapkeModel = get_hapke_model(phase_fn=phase_fn, scatter=scatter_type)
    thetai, thetae = np.deg2rad([float(thetai), float(thetae)])
    self.hapke_scalar = HapkeModel(thetai, thetae, float(n1), float(Bg))

    self.spectra = {}
    for key, infile in [('sml', small_file),
                        ('med', medium_file),
                        ('big', large_file),
                        ('file1', file1),
                        ('file2', file2),
                        ('file3', file3),
                        ('file4', file4),
                        ('file5', file5),
                        ('file6', file6),
                        ('file7', file7)]:
      #Checks if the infile variable has a string -- sanity check if not all ten files are uploaded
      if not infile == '':
        self.spectra[key] = analysis.loadmat_single(infile)

    # data to be filled in (later) for each grain size
    self.pp_spectra = {}
    self.ks = {}
    self.guesses = {}

    if self.hapke_scalar.needs_isow:
      # store the calibration spectrum
      specwave = analysis.loadmat_single(specwave_file).ravel()
      calspec = analysis.loadmat_single(calspec_file).ravel()
      self.calspec = np.column_stack((specwave, calspec))

    # plot the loaded spectra
    num_plots = 2 if self.hapke_scalar.needs_isow else 1
    fig = Figure(figsize=(9, 4), frameon=False, tight_layout=True)
    ax1 = fig.add_subplot(1, num_plots, 1)

    #Three default files loaded into the spectra dictionary are plotted on the graph
    ax1.plot(*self.spectra['sml'].T, label='Small grain')
    ax1.plot(*self.spectra['med'].T, label='Medium grain')
    ax1.plot(*self.spectra['big'].T, label='Large grain')
    
    #Adding plots for files uploaded - can upload maximum of 10 files including the three default
    for k in self.file_key_list:
      if k in self.spectra:
        ax1.plot(*self.spectra[k].T, label=k)

    ax1.set_xlabel('Wavelength ($\mu{}m)$')
    ax1.set_ylabel('Reflectance')
    ax1.legend(fontsize='small', loc='best')
    ax1.set_title('Input VNIR spectra')
    if self.hapke_scalar.needs_isow:
      ax2 = fig.add_subplot(1, num_plots, 2)
      ax2.plot(specwave, calspec, 'k-')
      ax2.set_title('Calibrated standard')
      ax2.set_xlabel('Wavelength ($\mu{}m)$')
    # return html for a plot of the inputs + calibration
    return 'Initialization complete.', None, [fig]

  #Corresponds to section 1 of Matlab code - Finding the lambda and fitting the polynomial curve
  def preprocess(self, low=0, high=0, UV=0, fit_order=0):
    low, high, UV = float(low), float(high), float(UV)
    self.pp_bounds = (low, high, UV)
    fit_order = int(fit_order)

    if self.hapke_scalar.needs_isow:
      # initialize isow as a scalar
      isoind1, isoind2 = np.searchsorted(self.calspec[:,0], (low, high))
      self.hapke_scalar.set_isow(self.calspec[isoind1:isoind2,1].mean())

    # run preprocessing on each spectrum
    for key, traj in self.spectra.items():
      self.pp_spectra[key] = analysis.preprocess_traj(traj, low, high, UV,
                                                      fit_order=fit_order)

    # plot the results
    fig = Figure(figsize=(6, 4), frameon=False, tight_layout=True)
    ax = fig.gca()
    ax.plot(*self.pp_spectra['sml'].T, label='Small grain')
    ax.plot(*self.pp_spectra['med'].T, label='Medium grain')
    ax.plot(*self.pp_spectra['big'].T, label='Large grain')

    #If additional files exist we plot them 
    for k in self.file_key_list:
      if k in self.pp_spectra:
        ax.plot(*self.pp_spectra[k].T, label=k)

    ax.legend(fontsize='small', loc='best')
    ax.set_title('Preprocessed spectra')
    ax.set_xlabel('Wavelength ($\mu{}m)$')
    ax.set_ylabel('Reflectance')
    #pp is the parameter used for identifying the download data.
    return 'Preprocessing complete: ', 'pp', [fig]

  #Section Two by Default, Section Three and Four - Matlab Code
  def solve_for_k(self, key='sml', b=0, c=0, ff=0, s=0, D=0):
    b, c, s, D, ff = map(float, (b, c, s, D, ff))
    self.guesses[key] = (b, c, s, D, ff)
    traj = self.pp_spectra[key]
    plt.close('all')  # hack!
    #The hidden treasure where all the brains are hidden
    self.ks[key] = analysis.MasterHapke1_PP(
        self.hapke_scalar, traj, b, c, ff, s, D, debug_plots=True)
    figures = [plt.figure(i) for i in plt.get_fignums()]
    return 'Solved for k: ', 'k-' + key, figures

  def get_loaded_samples(self):
    return list(self.pp_spectra.keys())

  def optimize_global_k(self, guess_key='med', opt_strategy='fast',
                        lowb1=0, lowb2=0, lowb3=0, upb1=0, upb2=0, upb3=0,
                        lowc1=0, lowc2=0, lowc3=0, upc1=0, upc2=0, upc3=0,
                        lows1=0, lows2=0, lows3=0, ups1=0, ups2=0, ups3=0,
                        lowD1=0, lowD2=0, lowD3=0, upD1=0, upD2=0, upD3=0,
                        lowk=0, upk=0, num_solns=1):
    self.hapke_vector_isow = self.hapke_scalar.copy()
    if self.hapke_vector_isow.needs_isow:
      # use vector isow, instead of the scalar we had before
      _, high, UV = self.pp_bounds
      idx1, idx2 = np.searchsorted(self.calspec[:,0], (UV, high))
      self.hapke_vector_isow.set_isow(self.calspec[idx1:idx2,1])

    # set up initial guesses
    k = self.ks[guess_key]
    guesses = np.empty(len(k) + 12)
    ff = np.zeros(3)
    for i, key in enumerate(('sml', 'med', 'big')):
      g = self.guesses[key]
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
    self.bounds = (lb, ub)

    # solve
    if opt_strategy == 'fast':
      solns = analysis.optimize_global_k(
          self.hapke_vector_isow, self.pp_spectra, guesses, lb, ub, ff,
          num_iters=int(num_solns))
      best_soln = solns[-1]
    else:
      tmp = analysis.MasterHapke2_PP(
          self.hapke_vector_isow, self.pp_spectra, guesses, lb, ub, ff,
          tr_solver='lsmr', verbose=2, spts=int(num_solns))
      solns = [res.x for res in tmp]
      best_soln = min(tmp, key=lambda res: res.cost).x

    # save the best solution
    self.ks['global'] = best_soln[12:]
    for i, key in enumerate(('sml', 'med', 'big')):
      b, c, s, D = best_soln[i:12:3]
      self.guesses[key] = (b, c, s, D, ff[i])

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
      wave, orig = self.pp_spectra[key].T
      b, c, s, D = best_soln[i:12:3]
      scat = self.hapke_vector_isow.scattering_efficiency(best_soln[12:], wave,
                                                          D, s)
      rc = self.hapke_vector_isow.radiance_coeff(scat, b, c, ff[i])

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
    ax.plot(wave, self.ks['sml'], label='Small')
    ax.plot(wave, self.ks['med'], label='Medium')
    ax.plot(wave, self.ks['big'], label='Large')
    ax.plot(wave, self.ks['global'], 'k--', label='Global')
    ax.set_xlabel('Wavelength (um)')
    ax.set_title('Fitted k')
    ax.legend(fontsize='small', loc='best')

    msg = 'Finished %d iterations: ' % len(solns)
    return msg, 'k-global', [fig1, fig2, fig3]

  def add_mir_data(self, mirk_file='', mirv_file=''):
    # HACK: use defaults if some/all files aren't provided
    # mirk_file = mirk_file or '../data/kjar_110813_disp_k.mat'
    # mirv_file = mirv_file or '../data/kjar_110813_disp_v.mat'
    mirk_file = mirk_file or '../data/bytMIRk.mat'
    mirv_file = mirv_file or '../data/bytMIRv.mat'

    wave = self.pp_spectra['med'][:,0]
    k = self.ks['global']
    disp = analysis.MasterKcombine(mirk_file, mirv_file, wave, k)
    self.dispersion = disp

    # plot the combined dispersion data
    fig, ax = plt.subplots(figsize=(6, 4), frameon=False)
    ax.plot(10000/disp[:,0], disp[:,1])
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('k')
    ax.set_title('Combined k')

    return 'Combined MIR + VNIR k: ', 'dispersion', [fig]

  def run_sskk(self, anchor=0, grid_size=100):
    v, n = analysis.MasterSSKK(self.dispersion, self.hapke_scalar.n,
                               float(anchor), num_intervals=int(grid_size)).T
    self.hapke_vector_n = self.hapke_vector_isow.copy(refraction_index=n)

    fig, ax = plt.subplots(figsize=(6, 4), frameon=False)
    ax.plot(10000/v, n)
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('n')
    ax.set_title('Index of Refraction (n)')

    return 'Solved for n: ', 'n', [fig]

  def solve_phase(self, phase_files=(), phase_thetai=(), phase_thetae=(),
                  phase_sizes=(), downsample_factor=1):
    # HACK: use defaults if inputs aren't provided
    if not phase_files:
      # phase_files = glob('../data/smKJ*.mat')
      phase_files = glob('../data/BytW*.asc')
      info = [os.path.splitext(os.path.basename(f))[0][4:] for f in phase_files]
      size_map = dict(S='sml', M='med', B='big')
      phase_sizes = [size_map[x[0]] for x in info]
      phase_thetai = [int(x.split('i',1)[1].split('e',1)[0]) for x in info]
      phase_thetae = [int(x.split('e',1)[1].strip('b')) for x in info]

    # organize data by grain size, and
    # preprocess the new data like we did with the old data
    low, high, UV = self.pp_bounds
    data = {'sml': [], 'med': [], 'big': []}
    thetai = {'sml': [], 'med': [], 'big': []}
    thetae = {'sml': [], 'med': [], 'big': []}
    for f,sz,ti,te in zip(phase_files, phase_sizes, phase_thetai, phase_thetae):
      traj = analysis.loadmat_single(f)
      traj = analysis.preprocess_traj(traj, *self.pp_bounds)
      data[sz].append(traj)
      thetai[sz].append(np.deg2rad(float(ti)))
      thetae[sz].append(np.deg2rad(float(te)))

    # prep the inputs to the optimization code
    lb, ub = self.bounds
    k = self.ks['global']
    guesses = np.empty(4 + len(k))
    guesses[4:] = k

    # optimize each grainsize separately
    for sz in ('sml', 'med', 'big'):
      b, c, s, D, ff = self.guesses[sz]
      guesses[:4] = (b, c, s, D)
      model = self.hapke_vector_n.copy(incident_angle=thetai[sz],
                                       emission_angle=thetae[sz])
      solns = analysis.MasterPhase1(model, data[sz], guesses, lb, ub, ff)
      best_soln = solns[-1]  # maybe?

      # save the best solution
      self.ks['final_'+sz] = best_soln[4:]
      b, c, s, D = best_soln[:4]
      self.guesses['final_'+sz] = (b, c, s, D, ff)

    return '', []

  #Download Handler - When param passed is according to the section
  def _download_data(self, param):
    names = {
        'sml': 'SmallGrain', 'med': 'MediumGrain', 'big': 'LargeGrain',
        'global': 'Global', 
        'file1':'File_1', 'file2': 'File_2','file3':'File_3', 'file4': 'File_4', 'file5':'File_5', 'file6': 'File_6', 'file7' : 'File_7'
    }
    if param == 'pp':
      buf = BytesIO()
      with ZipFile(buf, mode='w') as zf:
        for key in self.pp_spectra:
          fname = '%s.txt' % names[key]
          zf.writestr(fname, _traj2bytes(self.pp_spectra[key]))
      return 'preprocessed.zip', 'application/zip', buf.getvalue()
    elif param.startswith('k-'):
      key = param.split('-', 1)[1]
      fname = 'k_%s.txt' % names[key]
      print_vec = _vec2bytes(self.ks[key])
      return fname, 'text/plain', print_vec
    elif param == 'dispersion':
      return 'combined_k.txt', 'text/plain', _traj2bytes(self.dispersion)
    elif param == 'n':
      return 'n.txt', 'text/plain', _vec2bytes(self.hapke_vector_n.n)
    else:
      raise ValueError('Unknown download type: %r' % param)


def _traj2bytes(traj):
  return b'\n'.join(b'%r\t%r' % tuple(row) for row in traj)


def _vec2bytes(arr):
  return b'\n'.join(b'%r' % x for x in arr)