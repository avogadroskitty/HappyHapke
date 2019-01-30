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
  def initialize(self, phase_fn='legendre', scatter_type='lambertian',
                 thetai=0, thetae=0, n1=0, Bg=0,
                 specwave_file='', calspec_file='', **kwargs):  
    # HACK: use defaults if some/all files aren't provided
    specwave_file = specwave_file or '../data/specwave2.mat'
    calspec_file = calspec_file or '../data/calspecw2.mat'

    # initialize the model
    HapkeModel = get_hapke_model(phase_fn=phase_fn, scatter=scatter_type)
    thetai, thetae = np.deg2rad([float(thetai), float(thetae)])
    self.hapke_scalar = HapkeModel(thetai, thetae, float(n1), float(Bg))

    self.spectra = {}
    for key in kwargs:
        if 'file' in key:
      #Checks if the infile variable has a string -- sanity check if not all ten files are uploaded
      #self.spectra has the number of grain files included in the process
          if not kwargs[key] == '':
            self.spectra[key] = analysis.loadmat_single(kwargs[key])

    # data to be filled in (later) for each grain size
    self.pp_spectra = {}
    self.ks = {}
    self.guesses = {}
    self.scat_eff_grain = {}

    if self.hapke_scalar.needs_isow:
      # store the calibration spectrum
      specwave = analysis.loadmat_single(specwave_file).ravel()
      calspec = analysis.loadmat_single(calspec_file).ravel()
      self.calspec = np.column_stack((specwave, calspec))

    # plot the loaded spectra
    num_plots = 2 if self.hapke_scalar.needs_isow else 1
    fig = Figure(figsize=(9, 4), frameon=False, tight_layout=True)
    ax1 = fig.add_subplot(1, num_plots, 1)
    
    #Adding plots for files uploaded - can upload maximum of 10 files including the three default
    for k in self.spectra:
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
  def preprocess(self, low=0.32, high=2.55, UV=0, fit_order=1):
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

    #If additional files exist we plot them 
    for k in self.pp_spectra:
        ax.plot(*self.pp_spectra[k].T, label=k)

    ax.legend(fontsize='small', loc='best')
    ax.set_title('Preprocessed spectra')
    ax.set_xlabel('Wavelength ($\mu{}m)$')
    ax.set_ylabel('Reflectance')
    #pp is the parameter used for identifying the download data.
    return 'Preprocessing complete: ', 'pp', [fig]

  #Section Two by Default, Section Three and Four - Matlab Code
  def solve_for_k(self, key='file2', b=0, c=0, ff=0.000000001, s=0, D=0):
    b, c, s, D, ff = map(float, (b, c, s, D, ff))
    self.guesses[key] = (b, c, s, D, ff)
    traj = self.pp_spectra[key]
    plt.close('all')  # hack!
    #The hidden treasure where all the brains are hidden
    solved_k, scat_eff = analysis.MasterHapke1_PP(
        self.hapke_scalar, traj, b, c, ff, s, D, debug_plots=True)

    self.ks[key] = solved_k
    self.scat_eff_grain[key] = scat_eff
    figures = [plt.figure(i) for i in plt.get_fignums()]
    return 'Solved for k: ', 'sk-' + key, figures

  def optimize_global_k(self, guess_key='file2', opt_strategy='slow',lowbfile1=0, lowbfile2=0, lowbfile3=0, upbfile1=0, upbfile2=0, upbfile3=0,
                       lowcfile1=0, lowcfile2=0, lowcfile3=0, upcfile1=0, upcfile2=0, upcfile3=0, lowsfile1=0, lowsfile2=0, lowsfile3=0,
                      upsfile1=0, upsfile2=0, upsfile3=0,lowDfile1=0, lowDfile2=0, lowDfile3=0, upDfile1=0, upDfile2=0, upDfile3=0,lowk=0, upk=0, num_solns=1, **kwargs):
    #The previous step only approximates for a single grain size
    #Should we have guesses for all grain samples or only the ones we have approximated for?
    no_of_grain_samples = len(self.spectra)
    total_guesses = no_of_grain_samples * 4 # 4 values (b,c,s,D) for each grain size

    self.hapke_vector_isow = self.hapke_scalar.copy()
    if self.hapke_vector_isow.needs_isow:
      # use vector isow, instead of the scalar we had before
      _, high, UV = self.pp_bounds
      idx1, idx2 = np.searchsorted(self.calspec[:,0], (UV, high))
      self.hapke_vector_isow.set_isow(self.calspec[idx1:idx2,1])

    # set up initial guesses
    print(self.ks)
    k = self.ks[guess_key]
    # [215,] -- size of the array
    guesses = np.empty(len(k) + total_guesses)
    #[215 + (4 * no of grains),] - size of the guesses list
    ff = np.zeros(no_of_grain_samples)
    for i, key in enumerate(sorted(self.guesses.keys())):
      g = self.guesses[key]
      # Unpacks the b,c,s,D values for each grain size into one large array. g holds b,c,s,D,f -- we take only the first four
      guesses[i:total_guesses:no_of_grain_samples] = g[:4] 
      # guesses Example:
      # for sml, med anf big grain sizes
      # [sml-b, med-b, big-g, sml-C, med-C, big-C, sml-S, med-S, big-S, sml-D, med-D, big-D, 215 values of K]
      # total with the length of K - 4 values for each grain size -- this is the magic 12
      ff[i] = g[4]
    guesses[total_guesses:] = k #Filling the rest of the array with the value of K
    # set up bounds
    lb = np.empty_like(guesses)
    #Values that will be there regardless if additional grain sizes are uploaded
    lb[:12] = [lowbfile1, lowbfile2, lowbfile3, lowcfile1, lowcfile2, lowcfile3, lowsfile1, lowsfile2, lowsfile3,
    lowDfile1, lowDfile2, lowDfile3]

    temp_low_bound = []
    temp_up_bound=[]
    for grain in self.spectra.keys():
      if grain not in ['file1', 'file2', 'file3']:
        temp_low_bound.append(kwargs['lowb'+grain])
        temp_low_bound.append(kwargs['lowc'+grain])
        temp_low_bound.append(kwargs['lows'+grain])
        temp_low_bound.append(kwargs['lowD'+grain])

        temp_up_bound.append(kwargs['upb'+grain])
        temp_up_bound.append(kwargs['upc'+grain])
        temp_up_bound.append(kwargs['ups'+grain])
        temp_up_bound.append(kwargs['upD'+grain])

    lb[12:total_guesses] = temp_low_bound
    #Filling in rest of the values
    lb[total_guesses:] = lowk

    ub = np.empty_like(guesses)
    ub[:12] = [upbfile1, upbfile2, upbfile3, upcfile1, upcfile2, upcfile3, upsfile1, upsfile2, upsfile3,
    upDfile1, upDfile2, upDfile3]
    ub[12:total_guesses] = temp_up_bound
    ub[total_guesses:] = upk
    self.bounds = (lb, ub)

    # solve
    tmp = analysis.MasterHapke2_PP(
        self.hapke_vector_isow, self.pp_spectra, guesses, lb, ub, ff,
        tr_solver='lsmr', verbose=2, spts=int(num_solns))
    solns = [res.x for res in tmp]
    best_soln = min(tmp, key=lambda res: res.cost).x

    # save the best solution
    self.ks['global'] = best_soln[total_guesses:]
    for i, key in enumerate(sorted(self.spectra.keys())):
      b, c, s, D = best_soln[i:total_guesses:no_of_grain_samples]
      self.guesses[key] = (b, c, s, D, ff[i])

    # plot solved parameters (b, c, s, D) for each grain size
    fig1, axes = plt.subplots(figsize=(9,5), ncols=4, nrows=no_of_grain_samples, sharex=True,
                              frameon=False)

    #Take out for now - responisble for error 'builtin function or method object is not iterable
    #Label the rows
    #for i, key in enumerate(self.spectra.keys()):
    #  axes[i,0].set_ylabel(key)
    
    #Label the columns
    axes[0,0].set_ylabel('file1')
    axes[1,0].set_ylabel('file2')
    axes[2,0].set_ylabel('file3')
    axes[0,0].set_title('b')
    axes[0,1].set_title('c')
    axes[0,2].set_title('s')
    axes[0,3].set_title('D')
    for i, key in enumerate(sorted(self.spectra.keys())):
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
    for i, key in enumerate(sorted(self.spectra.keys())):
      wave, orig = self.pp_spectra[key].T
      b, c, s, D = best_soln[i:total_guesses:no_of_grain_samples]
      scat = self.hapke_vector_isow.scattering_efficiency(best_soln[total_guesses:], wave,
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
    for key in self.spectra.keys():
      ax.plot(wave, self.ks[key], label=key)

    ax.plot(wave, self.ks['global'], 'k--', label='Global')
    ax.set_xlabel('Wavelength (um)')
    ax.set_title('Fitted k')
    ax.legend(fontsize='small', loc='best')

    msg = 'Finished %d iterations: ' % len(solns)
    return msg, 'k-global', [fig1, fig2, fig3]

  def add_mir_data(self, mirk_file='', mirv_file='', adjType=3):
    mirk_file = mirk_file or '../data/bytMIRk.mat'
    mirv_file = mirv_file or '../data/bytMIRv.mat'

    wave = self.pp_spectra['file2'][:,0]
    k = self.ks['global']
    plt.close('all')  # hack!
    pltData, kset = analysis.MasterKcombine(mirk_file, mirv_file, wave, k, adjType)
    self.vnirv, self.vnirk, self.fullv, self.fullk = kset
    figures = [plt.figure(i) for i in plt.get_fignums()]
    self.mirdata = pltData
    
    return 'Combined MIR + VNIR k: ', 'mirdata', figures

  def run_sskk(self, anchor=0.58929, grid_size=100, iter=1):
    #This section determined the real index of refraction, n, from k, using a
    #singly subtractive Kramers Kronig calculation. For this lamdiff, you will
    #need MIR data of your sample and should obtain n and k for those data
    #using a dispersion analysis (see other material).

    #Select anchor wavelength (this is usually sodium D line of 0.58929um)
    #this is the wavelength at which n1 was determined
    #iteration through program

    n1 = self.hapke_scalar.n1
    kset = self.vnirv, self.vnirk, self.fullv, self.fullk

    wave = self.pp_spectra['file2'][:,0]
    _, lend, lstart = self.pp_bounds #low, high, UV

    pltData, vars = analysis.MasterSSKK(kset, anchor, iter, wave, n1, lstart, lend)

    figures = [plt.figure(i) for i in plt.get_fignums()]
    self.sskk = pltData

    ## Should this be overwritten in self.pp_bounds -- I guess not we need the low from preprocessing
    self.sskk_lstart, self.sskk_lend, self.sskk_lamdiff, self.vislam, self.visn = vars
    
    return 'Solved for n: ', 'sskk', figures

  def phase_solver(self, pfile1='', pfile2='', pfile3='', pfile4='', pfile5='', pfile6='', pfile7='',  pfile8='', pfile9='', pfile10='',
                   maxScale=10, lowb=0, upb=1, lowc=0, upc=1, lows1=0, ups1 = 0.06, lows2=0, ups2=0.06, lows3=0, ups3=0.06, maxfun = 1000000000000000000, spts=30, funtol = 0.00000000000001, xtol= 0.00000000000001, maxit=1000 , 
                   lowd1=21, upd1=106, lowd2=31, upd2=150, lowd3=50, upd3=180, guess_b=0.4, guess_c=0.8, guess_d1=50, guess_d2=90, guess_d3=140, guess_s1=0.06, guess_s2=0.04, guess_s3=0.02 ):
      k = self.ks['global']

      pfile1 = pfile1 or '../data/BytWS63106i30e0.asc' 
      pfile2 = pfile2 or '../data/BytWM106150i30e0.asc'
      pfile3 = pfile3 or '../data/BytWB150180i30e0.asc'

      #Input: grain size, phase angle

      #Change to include 70 --- yes 70 files : min, 6*3
      self.phase_file_key_list = ['pfile4','pfile5','pfile6','pfile7', 'pfile8','pfile9','pfile10']

      self.phases = {}
      for key, infile in [('pfile1', pfile1),
                        ('pfile2', pfile2),
                        ('pfile3', pfile3),
                        ('pfile4', pfile4),
                        ('pfile5', pfile5),
                        ('pfile6', pfile6),
                        ('pfile7', pfile7),
                        ('pfile8', pfile8),
                        ('pfile9', pfile9),
                        ('pfile10', pfile10)]:
        #Checks if the infile variable has a string -- sanity check if not all ten files are uploaded
        #self.phases has the number of grain files included in the process
        if not infile == '':
            self.phases[key] = analysis.loadmat_single(infile) # Shape (N,2)

        #This program will use data from multiple viewing geometries to calculate
        #phase function parameters for a sample where k and n for i=30, e=0 is already
        #known. 
        #This program downsamples the data and then uses a
        #minimization routine to find the best wavelength dependent b and c
        #coefficients for the phase function by minimizing the difference between
        #the calculated and observed data for multiple viewing geometries and
        #multiple grain sizes simultaneously.

      lstart2 = self.sskk_lstart
      lend2 = self.sskk_lend
      lamdiff = self.sskk_lamdiff
      low, high, UV = self.pp_bounds
      vislam, visn = self.vislam, self.visn
      wavelength = self.pp_spectra['file2'][:,0] 
      params = (lstart2, lend2, low, UV, lamdiff, maxScale, lowb, upb, lowc, upc, lows1, ups1, lows2, ups2, lows3, ups3, 
                   lowd1, upd1, lowd2, upd2, lowd3, upd3, guess_b, guess_c, guess_d1, guess_d2, guess_d3, guess_s1, guess_s2, guess_s3, 
                   maxfun, funtol, xtol, maxit, spts, vislam, visn, wavelength, k)

      pltdata, vars = analysis.solve_phase(self.phases, params)

      return 'Phase Solved ', 'psolve', figures

  def solve_phase(self, phase_files=(), phase_thetai=(), phase_thetae=(),
                  phase_sizes=(), downsample_factor=1):
    # HACK: use defaults if inputs aren't provided
    if not phase_files:
      # phase_files = glob('../data/smKJ*.mat')
      phase_files = glob('../data/BytW*.asc')
      info = [os.path.splitext(os.path.basename(f))[0][4:] for f in phase_files]
      size_map = dict(S='file1', M='file2', B='file3')
      phase_sizes = [size_map[x[0]] for x in info]
      phase_thetai = [int(x.split('i',1)[1].split('e',1)[0]) for x in info]
      phase_thetae = [int(x.split('e',1)[1].strip('b')) for x in info]

    # organize data by grain size, and
    # preprocess the new data like we did with the old data
    low, high, UV = self.pp_bounds
    data = {'file1': [], 'file2': [], 'file3': []}
    thetai = {'file1': [], 'file2': [], 'file3': []}
    thetae = {'file1': [], 'file2': [], 'file3': []}
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
    for sz in ('file1', 'file2', 'file3'):
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
    elif param.startswith('sk-'):
      key = param.split('-', 1)[1]
      buf = BytesIO()
      with ZipFile(buf, mode='w') as zf:
        for s_data in self.scat_eff_grain[key]: 
          file = '%s.txt' % s_data[0]
          zf.writestr(file, _plot2bytes(s_data[1], s_data[2]))
      return 'solved_k_data-%s.zip' % key, 'application/zip', buf.getvalue()
    elif param.startswith('k-'):
      key = param.split('-', 1)[1]
      fname = 'k_%s.txt' % names[key]
      print_vec = _vec2bytes(self.ks[key])
      return fname, 'text/plain', print_vec
    elif param == 'mirdata':
        buf = BytesIO()
        with ZipFile(buf, mode='w') as zf:
            for plt in self.mirdata:
                file = '%s.txt' % plt[0]
                zf.writestr(file, _plot2bytes(plt[1], plt[2]))
        return 'MIR Data.zip', 'application/zip', buf.getvalue()
    elif param == 'sskk':
        buf = BytesIO()
        with ZipFile(buf, mode='w') as zf:
            for plt in self.sskk:
                file = '%s.txt' % plt[0]
                zf.writestr(file, _plot2bytes(plt[1], plt[2]))
        return 'SSKK Data.zip', 'application/zip', buf.getvalue()
        #return 'n.txt', 'text/plain', _vec2bytes(self.hapke_vector_n.n)
    else:
      raise ValueError('Unknown download type: %r' % param)


def _traj2bytes(traj):
  return b'\n'.join(b'%r\t%r' % tuple(row) for row in traj)


def _vec2bytes(arr):
  return b'\n'.join(b'%r' % x for x in arr)

def _plot2bytes(x_data, y_data):
  return b'\n'.join(b'%r\t%r' % (x,y) for x,y in zip(x_data, y_data))
