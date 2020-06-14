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
from hapke_model import HapkeModel
from phaseangle import PhaseAngleObj
import pickle
  
class ProgramState(object):
  def initialize(self, phase_fn='legendre', scatter_type='lambertian',
                 thetai=0, thetae=0, n1=0, 
                 specwave_file='', calspec_file='', **kwargs):  
    
    plt.close('all')  # hack!
    # HACK: use defaults if some/all files aren't provided
    specwave_file = specwave_file or '../data/specwave2.mat'
    calspec_file = calspec_file or '../data/calspecw2.mat'
    self.Bg = True if 'Bg' in kwargs else False
    Hu = True if 'HuApprox' in kwargs else False
     
    self.global_thetai, self.global_thetae = np.deg2rad([float(thetai), float(thetae)])
    self.hapke_scalar = HapkeModel(self.global_thetai, self.global_thetae, float(n1), self.Bg, phase_fn, scatter_type, Hu)

    self.spectra = {}
    self.n1 = float(n1)
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

    self.tmp, self.scat, self.rc = None, {}, {}

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
  def preprocess(self, low=0.32, high=2.55, UV=0, fit_order=1, idx_in=3):
    
    plt.close('all')  # hack!
    low, high, UV = float(low), float(high), float(UV)
    self.pp_bounds = (low, high, UV)
    fit_order = int(fit_order)
    idx_in = int(idx_in)

    if self.hapke_scalar.needs_isow:
      # initialize isow as a scalar
      isoind1, isoind2 = np.searchsorted(self.calspec[:,0], (low, high))
      self.hapke_scalar.set_isow(self.calspec[isoind1:isoind2,1].mean())

    # run preprocessing on each spectrum
    for key, traj in self.spectra.items():
      self.pp_spectra[key] = analysis.preprocess_traj(traj, low, high, UV,
                                                      fit_order=fit_order, idx = idx_in)

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

  def solve_for_all_k(self, **kwargs):
    
    plt.close('all')  # hack!
    b0 = None
    h = None
    for i, key in enumerate(sorted(self.pp_spectra)):
        for k,v in kwargs.items():
            if key in k.lower():
                if 'b_' in k:
                    b = v
                elif 'c' in k:
                    c = v
                elif 's' in k:
                    s = v
                elif 'ff' in k:
                    ff = v
                elif 'D' in k:
                    D = v
                elif 'b0' in k:
                    b0 = v
                elif 'h' in k:
                    h = v;
        
        if self.Bg:
            b, c, s, D, b0, h, ff = map(float, (b, c, s, D, b0, h, ff))
            self.guesses[key] = b, c, s, D, b0, h, ff 
            self.valcnt = 6
        else:
            b, c, s, D, ff = map(float, (b, c, s, D, ff))
            self.guesses[key] = b, c, s, D, ff
            self.valcnt = 4
        
        traj = self.pp_spectra[key]
        #The hidden treasure where all the brains are hidden
        solved_k, scat_eff = analysis.MasterHapke1_PP(
            self.hapke_scalar, traj, b, c, ff, s, D, key, self.n1, debug_plots=True, b0=b0, h=h)

        self.ks[key] = solved_k
        self.scat_eff_grain[key] = scat_eff
    
    figures = [plt.figure(i) for i in plt.get_fignums()]
    return 'Solved for k: ', 'guessk', figures

  def optimize_global_k(self, guess_key='file2', lowk=0, upk=0, maxfun = 1000, 
                   spts=30, diff_step = 0.0001, funtol = 0.00000000000001, xtol= 0.00000000000001, **kwargs):
    
    plt.close('all')  # hack!
    #The previous step only approximates for a single grain size
    #Should we have guesses for all grain samples or only the ones we have approximated for?
    no_of_grain_samples = len(self.spectra)
    total_guesses = no_of_grain_samples * self.valcnt # 4 values (b,c,s,D) for each grain size or 6 values (b,c,s,d,b0,h)

    self.hapke_vector_isow = self.hapke_scalar.copy()
    if self.hapke_vector_isow.needs_isow:
      # use vector isow, instead of the scalar we had before
      _, high, UV = self.pp_bounds
      idx1, idx2 = np.searchsorted(self.calspec[:,0], (UV, high))
      self.hapke_vector_isow.set_isow(self.calspec[idx1:idx2,1])

    # set up initial guesses
    k = self.ks[guess_key]
    # [215,] -- size of the array
    guesses = np.empty(len(k) + total_guesses)
    #[215 + (4 * no of grains),] - size of the guesses list
    ff = np.zeros(no_of_grain_samples)
    for i, key in enumerate(sorted(self.guesses.keys())):
      g = self.guesses[key]
      # Unpacks the b,c,s,D values for each grain size into one large array. g holds b,c,s,D,f -- we take only the first four
      guesses[i:total_guesses:no_of_grain_samples] = g[:self.valcnt] 
      # guesses Example:
      # for sml, med anf big grain sizes
      # [sml-b, med-b, big-g, sml-C, med-C, big-C, sml-S, med-S, big-S, sml-D, med-D, big-D, 215 values of K]
      # total with the length of K - 4 values for each grain size -- this is the magic 12
      ff[i] = g[self.valcnt]
    guesses[total_guesses:] = k #Filling the rest of the array with the value of K
    # set up bounds
    lb = np.empty_like(guesses)
    ub = np.empty_like(guesses)

    for i,grain in enumerate(sorted(self.spectra.keys())):
        if self.valcnt == 6:
            lb[i:total_guesses:no_of_grain_samples] = (kwargs['lowb'+grain], kwargs['lowc'+grain], kwargs['lows'+grain], kwargs['lowD'+grain], kwargs['lowb0'+grain], kwargs['lowh'+grain])
            ub[i:total_guesses:no_of_grain_samples] = (kwargs['upb'+grain], kwargs['upc'+grain], kwargs['ups'+grain], kwargs['upD'+grain], kwargs['upb0'+grain], kwargs['uph'+grain])
        else: #we have no Bg
            lb[i:total_guesses:no_of_grain_samples] = (kwargs['lowb'+grain], kwargs['lowc'+grain], kwargs['lows'+grain], kwargs['lowD'+grain])
            ub[i:total_guesses:no_of_grain_samples] = (kwargs['upb'+grain], kwargs['upc'+grain], kwargs['ups'+grain], kwargs['upD'+grain])
        
        
    #Filling in rest of the values
    lb[total_guesses:] = lowk
    ub[total_guesses:] = upk

    self.bounds = (lb, ub)
    self.global_ff = ff
    # solve
    self.tmp = analysis.MasterHapke2_PP(
        self.hapke_vector_isow, self.pp_spectra, guesses, lb, ub, ff, self.n1, self.valcnt,
         int(spts), int(maxfun),  float(diff_step), float(funtol), float(xtol), tr_solver='lsmr', verbose=2)
    solns = [res.x for res in self.tmp]
    best_soln = min(self.tmp, key=lambda res: res.cost).x

    # save the best solution
    self.ks['global'] = best_soln[total_guesses:]
    for i, key in enumerate(sorted(self.spectra.keys())):
      if self.Bg:
          b, c, s, D, b0, h = best_soln[i:total_guesses:no_of_grain_samples]
          self.guesses[key] = (b, c, s, D, b0, h, ff[i])
      else:
          b, c, s, D = best_soln[i:total_guesses:no_of_grain_samples]
          self.guesses[key] = (b, c, s, D, ff[i])
      

    # plot solved parameters (b, c, s, D) for each grain size
    fig1, axes = plt.subplots(figsize=(9,5), ncols=self.valcnt, nrows=no_of_grain_samples, sharex=True,
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
    if self.valcnt == 6:
        axes[0,4].set_title('b0')
        axes[0,5].set_title('h')
    for i, key in enumerate(sorted(self.spectra.keys())):
      for j in range(self.valcnt):
        ax = axes[i,j]
        idx = i + j*no_of_grain_samples
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
    #best_soln = solns[-1]
    for i, key in enumerate(sorted(self.spectra.keys())):
      wave, orig = self.pp_spectra[key].T
      b0 = None
      h = None
      if self.Bg:
          b, c, s, D, b0, h = best_soln[i:total_guesses:no_of_grain_samples]
      else:
          b, c, s, D = best_soln[i:total_guesses:no_of_grain_samples]
      self.scat[key] = self.hapke_vector_isow.scattering_efficiency(best_soln[total_guesses:], wave,
                                                          D, s, self.n1)
      self.rc[key] = self.hapke_vector_isow.radiance_coeff(self.scat[key], b, c, ff[i], b0, h)

      ax1.plot(wave, orig,label=('%s grain' % key))
      ax1.plot(wave, self.rc[key], 'k--')
      ax1.set_xlabel('Wavelength (um)')
      ax1.set_ylabel('Reflectance (#)')
      ax1.set_title('Final fit')
      ax1.legend(fontsize='small', loc='best')

      ax2.plot(wave, np.abs(self.rc[key] - orig), lw=1,
               label=('%s fit' % key))
      ax2.set_title('Fit error')
      ax2.set_xlabel('Wavelength (um)')
      ax2.set_ylabel('Abs. Error')
      ax2.legend(fontsize='small', loc='best')

    # plot original ks vs global k
    fig3, ax = plt.subplots(figsize=(6, 4), frameon=False)
    for key in self.spectra.keys():
      ax.plot(wave, self.ks[key], label=key)

    ax.semilogy(wave, self.ks['global'], 'k--', label='Global')
    ax.set_xlabel('Wavelength (um)')
    ax.set_title('Fitted k')
    ax.legend(fontsize='small', loc='best')

    msg = 'Finished %d iterations: ' % len(solns)
    return msg, 'k-global', [fig1, fig2, fig3] # The Download param for this section

  def add_mir_data(self, mirk_file='', mirv_file='', repk_file='', adjType=3):
    mirk_file = mirk_file or '../data/bytMIRk.mat'
    mirv_file = mirv_file or '../data/bytMIRv.mat'
    k = analysis.loadmat_single(repk_file)[:,1] if repk_file else self.ks['global']
    self.ks['global'] = k # if from file we over-write it otherwise this is redundant.

    wave = self.pp_spectra['file2'][:,0]
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
    plt.close('all')  # hack!

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

  def phase_solver(self, phaseAngleCount, fit_order, minScale, maxScale, minOffset, maxOffset, maxfun = 1000, 
                   spts=30, diff_step = 0.0001, funtol = 0.00000000000001, xtol= 0.00000000000001, **kwargs ):
      
      plt.close('all')  # hack!
      k = self.ks['global']
      
      #Input: grain size, phase angle
      self.phases = {}
      #Total no of grain sizes == no of pp_spectras
      no_grain_sizes = len(self.pp_spectra.keys())
      phaseGrainList = {}
      phase_bcsd = {}
      for i in range(no_grain_sizes):
          phaseGrainList[i] = []
          ii = str(i)
          if self.Bg:
              bcsd = float(kwargs['p_b_'+ii]), float(kwargs['p_c_'+ii]), float(kwargs['p_s_'+ii]), float(kwargs['p_d_'+ii]), float(kwargs['p_b0_'+ii]), float(kwargs['p_h_'+ii])
              lb_bcsd = float(kwargs['plb_b_'+ii]), float(kwargs['plb_c_'+ii]), float(kwargs['plb_s_'+ii]), float(kwargs['plb_d_'+ii]), float(kwargs['plb_b0_'+ii]), float(kwargs['plb_h_'+ii])              
              ub_bcsd = float(kwargs['pub_b_'+ii]), float(kwargs['pub_c_'+ii]), float(kwargs['pub_s_'+ii]), float(kwargs['pub_d_'+ii]), float(kwargs['pub_b0_'+ii]), float(kwargs['pub_h_'+ii])              
          else:
              bcsd = float(kwargs['p_b_'+ii]), float(kwargs['p_c_'+ii]), float(kwargs['p_s_'+ii]), float(kwargs['p_d_'+ii])
              lb_bcsd = float(kwargs['plb_b_'+ii]), float(kwargs['plb_c_'+ii]), float(kwargs['plb_s_'+ii]), float(kwargs['plb_d_'+ii])              
              ub_bcsd = float(kwargs['pub_b_'+ii]), float(kwargs['pub_c_'+ii]), float(kwargs['pub_s_'+ii]), float(kwargs['pub_d_'+ii])

          phase_bcsd[i] = bcsd, lb_bcsd, ub_bcsd
          for j in range(int(phaseAngleCount)):
              id = '_%s_%s' % (i,j)
              data = analysis.loadmat_single(kwargs['filepfile'+id])
              phaseAngle = PhaseAngleObj(i, kwargs['pfile_i'+id], kwargs['pfile_e'+id], data)
              phaseGrainList[i].append(phaseAngle)
          phaseGrainList[i].sort(key=lambda x: (x.incident_angle, x.emission_angle))
      
    #This program will use data from multiple viewing geometries to calculate
    #phase function parameters for a sample where k and n for i=30, e=0 is already
    #known. 
    #This program downsamples the data and then uses a
    #minimization routine to find the best wavelength dependent b and c
    #coefficients for the phase function by minimizing the difference between
    #the calculated and observed data for multiple viewing geometries and
    #multiple grain sizes simultaneously.
      ffs = self.global_ff

      lstart2 = self.sskk_lstart
      lend2 = self.sskk_lend
      lamdiff = self.sskk_lamdiff 
      low, high, UV = self.pp_bounds
      vislam, visn = self.vislam, self.visn
      wavelength = self.pp_spectra['file2'][:,0] 
      params = (lstart2, lend2, low, UV, lamdiff, float(minScale), float(maxScale), float(minOffset), float(maxOffset), int(maxfun), float(funtol), float(xtol), int(spts), float(diff_step),
                vislam, visn, wavelength, k, int(fit_order), int(phaseAngleCount), phaseGrainList, phase_bcsd, ffs, self.hapke_vector_isow)

      plt_data, allbest = analysis.solve_phase(self.phases, params)
      
      self.phase_best_soln, self.phase_bscale, self.phase_boffset, self.phase_k, self.phase_wave, self.phase_n = allbest
      figures = [plt.figure(i) for i in plt.get_fignums()]
      self.phase = plt_data

      return 'Phase Solved ', 'psolve', figures
  
  def repeat_k(self, lowk=0, upk=0, maxfun = 1000, spts=30, diff_step = 0.0001, funtol = 0.00000000000001, xtol= 0.00000000000001, **kwargs):
      plt.close('all')  # hack!    

      sizep = len(self.phase_k)
      grain_samples = len(self.pp_spectra.keys())
      gsvals = self.valcnt + 2
      total_guesses = grain_samples * (gsvals) 
      # gs3 == sb1,ob1,sc1,oc1,s1,D1,b0_1,h1 | sb2,ob2,sc2,oc2,s2,D2,b0_2,h2 | sb3,ob3,sc3,oc3,s3,D3,b0_3,h3
      rep_bounds = {}
      for i,grain in enumerate(sorted(self.spectra.keys())):
          if self.Bg:
              lb_bcsd = float(kwargs['scalelowb'+grain]), float(kwargs['offlowb'+grain]), float(kwargs['scalelowc'+grain]), float(kwargs['offlowc'+grain]), float(kwargs['lows'+grain]), float(kwargs['lowD'+grain]), float(kwargs['lowb0'+grain]), float(kwargs['lowh'+grain])
              ub_bcsd = float(kwargs['scaleupb'+grain]), float(kwargs['offupb'+grain]), float(kwargs['scaleupc'+grain]), float(kwargs['offupc'+grain]),  float(kwargs['ups'+grain]), float(kwargs['upD'+grain]), float(kwargs['upb0'+grain]), float(kwargs['uph'+grain]) 
          else:
              lb_bcsd = float(kwargs['scalelowb'+grain]), float(kwargs['offlowb'+grain]), float(kwargs['scalelowc'+grain]), float(kwargs['offlowc'+grain]), float(kwargs['lows'+grain]), float(kwargs['lowD'+grain])
              ub_bcsd = float(kwargs['scaleupb'+grain]), float(kwargs['offupb'+grain]), float(kwargs['scaleupc'+grain]), float(kwargs['offupc'+grain]),  float(kwargs['ups'+grain]), float(kwargs['upD'+grain])

          rep_bounds[i] = lb_bcsd, ub_bcsd

      if self.hapke_vector_isow.needs_isow:
          # use vector isow, instead of the scalar we had before
          _, high, UV = self.pp_bounds 
          idx1, idx2 = np.searchsorted(self.calspec[:,0], (UV, high))
          self.hapke_vector_isow.set_isow(self.calspec[idx1:idx2,1])

      guesses = np.empty(sizep + total_guesses)  
      bestsol = self.phase_best_soln[2:]
      prev_b = bestsol[:sizep*grain_samples]
      prev_c = bestsol[sizep*grain_samples:sizep*grain_samples*2]
      prev_s = bestsol[sizep*grain_samples*2:sizep*grain_samples*2+grain_samples]
      prev_D = bestsol[sizep*grain_samples*2+grain_samples:sizep*grain_samples*2+(2*grain_samples)]

      if self.Bg:
          prev_b0 = bestsol[sizep*grain_samples*2+(2*grain_samples):sizep*grain_samples*2+(3*grain_samples)]
          prev_h = bestsol[sizep*grain_samples*2+(3*grain_samples):]
      
      # scale default guess: 1, offset default guess: 0
      # i = 0,1,2
      for i,grain in enumerate(sorted(self.spectra.keys())):
          # 0 -8, 8 - 16, 16-24
          if self.Bg:
              guesses[i*gsvals: (i*gsvals)+gsvals] = np.array([1,0,1,0,prev_s[i], prev_D[i], prev_b0[i], prev_h[i]]) 
          else:
              guesses[i*gsvals: (i*gsvals)+gsvals] = np.array([1,0,1,0,prev_s[i], prev_D[i]])
      
      guess_k = self.phase_bscale * self.phase_k + self.phase_boffset
      guesses[total_guesses:] = guess_k

      # set up bounds    
      lb = np.empty_like(guesses)
      ub = np.empty_like(guesses)
      
      for i,grain in enumerate(sorted(self.spectra.keys())):
          lb[i*gsvals: (i*gsvals)+gsvals] = rep_bounds[i][0]
          ub[i*gsvals: (i*gsvals)+gsvals] = rep_bounds[i][1]
             
      lb[total_guesses:] = lowk 
      ub[total_guesses:] = upk

      params = (self.hapke_vector_isow, self.pp_spectra, self.global_thetai, self.global_thetae, guesses, prev_b, prev_c, lb, ub, self.global_ff, self.phase_n, self.phase_k, self.phase_wave, grain_samples, gsvals, total_guesses, int(spts), float(maxfun), float(diff_step), float(funtol), float(xtol))
      plt_data = analysis.Hapke_mastermind(params)
      figures = [plt.figure(i) for i in plt.get_fignums()]
      self.repk = plt_data

      return 'Phase Solved ', 'repk', figures
    
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
    elif param.startswith('k-'): # Global k step
      #key = param.split('-', 1)[1]
      #fname = 'k_%s.txt' % names[key]
      #print_vec = _vec2bytes(self.ks[key])
      #return fname, 'text/plain', print_vec

      ############################################
      ## June 14 2020 - Identified we need more data downloaded from global-k
      ############################################

      buf = BytesIO()
      with ZipFile(buf, mode='w') as zf:
        solns = [res.x for res in self.tmp]
        for i, resx in enumerate(solns): # All the solutions
            file = 'soln_%s.txt' % (str(i))
            zf.writestr(file, _vec2bytes(resx))

        ls_resx_cost = [res.cost for res in self.tmp]
        costfile = "all_solns_cost_file.txt"
        zf.writestr(costfile, _vec2bytes(ls_resx_cost))

        for i, key in enumerate(sorted(self.spectra.keys())):
            wave, orig = self.pp_spectra[key].T
            scat_grain = self.scat[key]
            rc_grain = self.rc[key]
            orig_file = 'orig_%s.txt' % (key)
            zf.writestr(orig_file, _plot2bytes(wave, orig))
            rc_file = 'rc_%s.txt' % (key)
            zf.writestr(rc_file, _plot2bytes(wave, rc_grain))
            scat_file = 'scat_%s.txt' % (key)
            zf.writestr(scat_file, _plot2bytes(wave, scat_grain))

      return 'global_k_data.zip', 'application/zip', buf.getvalue()

    elif param == 'guessk': #For all individual ks
        buf = BytesIO()
        with ZipFile(buf, mode='w') as zf:
            for key in self.scat_eff_grain.keys():
                for s_data in self.scat_eff_grain[key]:
                    file = '%s_%s.txt' % (s_data[0], key)
                    zf.writestr(file, _plot2bytes(s_data[1], s_data[2]))
        return 'solved_k_data.zip', 'application/zip', buf.getvalue()
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
    elif param == 'psolve':
        buf = BytesIO()
        with ZipFile(buf, mode='w') as zf:
            for plt in self.phase:
                file = '%s.txt' % plt[0]
                zf.writestr(file, _plot2bytes(plt[1], plt[2]))
        return 'PhaseSolve.zip', 'application/zip', buf.getvalue()
    elif param == 'repk':
        buf = BytesIO()
        with ZipFile(buf, mode='w') as zf:
            for plt in self.repk:
                file = '%s.txt' % plt[0]
                zf.writestr(file, _plot2bytes(plt[1], plt[2]))
        return 'RepeatK.zip', 'application/zip', buf.getvalue()
    else:
      raise ValueError('Unknown download type: %r' % param)
     
         
def _traj2bytes(traj):
  return b'\n'.join(b'%r\t%r' % tuple(row) for row in traj)


def _vec2bytes(arr):
  return b'\n'.join(b'%r' % x for x in arr)

def _plot2bytes(x_data, y_data):
  return b'\n'.join(b'%r\t%r' % (x,y) for x,y in zip(x_data, y_data))
   