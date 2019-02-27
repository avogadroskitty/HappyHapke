from __future__ import division, print_function
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
#Fix for Error in Section 3 - Solve For K in external Library Matplotlib
plt.switch_backend('agg')
from scipy.optimize import least_squares
from scipy.integrate import trapz
from scipy.interpolate import PchipInterpolator
from scipy import interpolate
from scipy import arange, array, exp
import math

def loadmat_single(path_or_file_obj):
  """Loads one array from a MAT-file or ascii file."""
  try:
    return np.loadtxt(path_or_file_obj)
  except ValueError:
    # must be a .mat file, so rewind to try again
    if hasattr(path_or_file_obj, 'seek'):
      path_or_file_obj.seek(0)
  mat = scipy.io.loadmat(path_or_file_obj, appendmat=False)
  keys = [k for k in mat.keys() if not k.startswith('_')]
  if len(keys) != 1:
    raise ValueError('loadmat_single expects one variable, got %r' % keys)
  return mat[keys[0]]


def prepare_spectrum(wavelength, spec, low, high):
  # make sure wavelength is in microns
  if wavelength[0] > 100:
      wavelength /= 1000

  # extract usable data
  idx1, idx2 = np.searchsorted(wavelength, (low, high))
  return wavelength[idx1:idx2 + 1], spec[idx1:idx2 + 1]


def fit_left_side(wavelength, spec, UV, fit_order=0, idx = 3):
  # fit the left end of the data with a polynomial
  lamdiff = wavelength[1] - wavelength[0]
  leftw = np.arange(UV, wavelength[0], lamdiff)
  if fit_order == 0:
    # extrapolate left side of k to 0.2 using ahrekiel's constant method
    # Cloutis 2008b shows that jarosite is essentially flat from 200-400nm
    padding = np.zeros_like(leftw)
    spec = np.concatenate((padding + spec[0], spec))
  else:
    # define end of curve to fit
    fit_order = min(fit_order, len(leftw))
    # fit curves
    # Takes the wavelength, spectrum points as x,y and fits a polynomial on it. Degree is the fitorder variable 
    # poly1d makes a one-d polynomial from the co-efficients returned by the polyfit method 
    fcoef = np.polyfit(wavelength[:idx], spec[:idx], fit_order)
    evalfn = evalPoly(fcoef, leftw)
    spec = np.concatenate([evalfn, spec])
  wavelength = np.concatenate((leftw, wavelength))
  return np.column_stack((wavelength, spec))


def preprocess_traj(traj, low, high, UV, fit_order=0, idx = 3):
  wave, spec = traj.T
  wave, spec = prepare_spectrum(wave, spec, low, high)
  return fit_left_side(wave, spec, UV, fit_order=fit_order, idx=idx)

#Solving for K - Logic -- setup the matrices here. For plotting get values from the hapke object - defined in hapke_model.py
def MasterHapke1_PP(hapke, traj, b, c, ff, s, D, key, n, debug_plots=False, b0 = None, h = None):
  wavelength, reflect = traj.T
  table_size = len(traj) * 2

  # make sure we have a scalar isow
  if hapke.needs_isow:
    assert np.isscalar(hapke.isow), 'MasterHapke1_PP requires scalar isow'

  # create table of increasing w (single scattering albedo) and use linear
  # interpolation to solve backwards from the real reflectance data
  w = np.linspace(0, 1, table_size, endpoint=False)
  rc = hapke.radiance_coeff(w, b, c, ff, b0, h)
  w2 = np.interp(reflect, rc, w)

  # use the same trick to back-solve for k from w2, except we have to plug in
  # the (k/wavelength) term when interpolating
  # TODO: incorporate bounds on k to determine table bounds
  k_wave = np.logspace(-1, -7, table_size)
  scat = hapke.scattering_efficiency(k_wave, 1, D, s, n)
  k_wave2 = np.interp(w2, scat, k_wave)
  k = k_wave2 * wavelength

  if debug_plots:
    # calculate scattering efficiency for each solved k
    rc2 = hapke.radiance_coeff(w2, b, c, ff, b0, h)
    ScatAlb = hapke.scattering_efficiency(k, wavelength, D, s, n)
    rc3 = hapke.radiance_coeff(ScatAlb, b, c, ff, b0, h)

    #The _ is the figure and the axes object is stores in axes
    fig, axes = plt.subplots(figsize=(10,4), nrows=2, ncols=2, sharex=True)
    # plot reflectance data and rc2 for comparison
    fig.suptitle('Graphs for %s' % key)
    ax = axes[0,0] #Position of the plots
    ax.plot(wavelength, reflect)
    ax.plot(wavelength, rc2, '--')# Third argument is for dotted lines
    ax.set_ylabel('Reflectance (#)')
    ax.set_title('Fit #1: scattering')
    ax2 = ax.twinx() # Makes another invisible copy of the X Axis -- that is shared from the previous plot
    ax2.plot(wavelength, np.abs(rc2 - reflect), 'k-', lw=1, alpha=0.75) #arguements are x data, y data, black solid line, line width and opacity
    ax2.set_ylabel('Fit Error')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

    # plot w2 and single scattering albedo
    ax = axes[0,1]
    ax.plot(wavelength, w2)
    ax.plot(wavelength, ScatAlb, '--')
    ax.set_title('Fit #2: k')
    ax.set_ylabel('Scattering Albedo')
    ax2 = ax.twinx()
    ax2.plot(wavelength, np.abs(w2 - ScatAlb), 'k-', lw=1, alpha=0.75)
    ax2.set_ylabel('Fit Error')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

    # plot reflectance vs rc3
    ax = axes[1,0]
    ax.plot(wavelength, reflect)
    ax.plot(wavelength, rc3, '--')
    ax.set_ylabel('Reflectance (#)')
    ax.set_xlabel('Wavelength (um)')
    ax.set_title('Combined fit')
    ax2 = ax.twinx()
    ax2.plot(wavelength, np.abs(rc3 - reflect), 'k-', lw=1, alpha=0.75)
    ax2.set_ylabel('Fit Error')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

    # plot fitted k
    ax = axes[1,1]
    ax.semilogy(wavelength, k) #Plot the graph in Log Scaling
    ax.set_ylabel('fitted k')
    ax.set_xlabel('Wavelength (um)')

    #A list for all the data that is plotted
    scat_eff_for_k = [ 
                        ['Wavelength Vs Reflect',wavelength, reflect],
                        ['Wavelength Vs Rc2', wavelength, rc2],
                        ['Wavelength Vs Rc2-Reflect', wavelength, np.abs(rc2 - reflect)],
                        ['Wavelength Vs W2', wavelength, w2], 
                        ['Wavelength Vs Scat Alb', wavelength, ScatAlb],
                        ['Wavelength Vs W2-ScatAlb', wavelength, np.abs(w2 - ScatAlb)],
                        ['Wavelength Vs Rc3', wavelength, rc3],
                        ['Wavelength Vs Rc3-Reflect', wavelength, np.abs(rc3 - reflect)],
                        ['Wavelength Vs K', wavelength, k]
                     ]
  return k, scat_eff_for_k

#Find what is magic 12
def MasterHapke2_PP(hapke, spectra, coefg, lb, ub, ff, n, valcnt,  spts, maxfun, diff_step, funtol, xtol, **kwargs):
  """This program performs an iterative minimization using Hapke's radiative
  transfer theory to find a global and grain-size independent value of
  imaginary index of refraction, k."""
  no_of_grain_samples = len(spectra)
  total_guesses = no_of_grain_samples * valcnt # 4 values (b,c,s,D) for each grain size

  wave = spectra['file2'][:,0]
  actuals = [spectra[key][:,1] for key in sorted(spectra.keys())]

  #Why 3 
  def obj_fn(coef):
    k = coef[total_guesses:]
    loss = 0
    for i, actual in enumerate(actuals):
      b0 = None
      h = None
      if hapke.needs_bg:
          b, c, s, D, b0, h = coef[i:total_guesses:no_of_grain_samples]
      else:
          b, c, s, D = coef[i:total_guesses:no_of_grain_samples]
       #these are paired correctly now that it is sorted elsewheres
      scat = hapke.scattering_efficiency(k, wave, D, s, n)
      rc = hapke.radiance_coeff(scat, b, c, ff[i], b0, h)
      loss += ((rc - actual)**2).sum()
    return np.sqrt(loss)

  start_points = np.empty((spts, len(coefg)))
  start_points[0] = coefg

  #Initialize random start points
  for i in range(1, spts):
    start_points[i] = np.random.uniform(lb, ub)

  #Start Points is 215 + 4 * no of grain samples

  bounds = np.row_stack((lb, ub))

  solutions = []
  #For each start point - minimize the least square error and append it to solutions.
  for spt in start_points:
    res = least_squares(obj_fn, spt, bounds=bounds, ftol=funtol, xtol=xtol,x_scale = 'jac', method='trf', max_nfev=maxfun, diff_step=diff_step, **kwargs)
    #res = least_squares(obj_fn, spt, bounds=bounds, method='trf', **kwargs)
    solutions.append(res)
  return solutions

##Used in Section Addd MIR Data
def MasterKcombine(mir_k, mir_v, wavelength, global_k, adjType):

    if adjType == '4':
        return no_mir_data(wavelength, global_k)

    k = loadmat_single(mir_k).ravel()
    v = loadmat_single(mir_v).ravel()
    
    #as iof MaskerKcombine_Bas.mat -- we need to plot the k and v
    _, axes = plt.subplots(figsize=(12,6), nrows=2, ncols=2)
    ax1 = axes[0,0]
    ax1.plot(v,k)
    ax1.set_ylabel('k')
    ax1.set_xlabel('Wavenumber (cm-1)')
    ax1.set_title('MIR k')
    ax1.invert_xaxis()

    #If you are changing anything here please change in no_mir_data()
    lam = wavelength
    vnirk = global_k

    #Plot vnirk vs wavelength -- 
    #If you are changing anything here please change in no_mir_data() where the same is plotted
    ax2 = axes[0,1]
    ax2.plot(lam, vnirk)
    ax2.set_ylabel('k')
    ax2.set_xlabel('Wavelength(um)')
    ax2.set_title('VNIR k')

    # a little modeling magic to make the arrays meet
    # in case you cut off some noise at the end

    # get a frequency space vector for vnirk
    #If you are changing anything here please change in no_mir_data()
    vnirv = 10000 / lam

    #Reverse only if not in ascending
    if vnirv[-1] < vnirv[0]:
        rev_vnirv = np.flip(vnirv, axis=0)
        rev_vnirk = np.flip(vnirk, axis=0)

    #If you are changing anything here please change in no_mir_data() where the same is plotted
    ax3 = axes[1,0]
    ax3.semilogy(vnirv, vnirk)
    ax3.set_ylabel('k')
    ax3.set_xlabel('Wavenumber(cm e-1)')
    ax3.set_title('Wavenumber(cm e-1) vs K')
    ax3.invert_xaxis() 

    #Evenly Spacing stuff
    vdiff = abs(v[1] - v[0])

    #Determine # of pts in VNIRV
    vnirvsize = round(abs(rev_vnirv[0] - rev_vnirv[-1])) / vdiff
    vnirvdiff =(abs(vnirv[0]-vnirv[-1]))/vnirvsize; 
    even_vnirv = np.linspace(rev_vnirv[0], rev_vnirv[-1], vnirvsize)
    interp_ob = PchipInterpolator(rev_vnirv, rev_vnirk)
    even_vnirk = interp_ob(even_vnirv)
    #This now has the same point spacing as v, so we should have same weight in integral

    #Taking out the noise
    noisy_end = max(lam)
    vals = abs(even_vnirv - 10000/noisy_end)
    minloc = np.argmin(vals) + 1 # Remove the noisy end
    new_vnirv = even_vnirv[minloc:] #even vnirv - is in ascending order -- so we need to get from minloc
    new_vnirk = even_vnirk[minloc:]

    ax4 = axes[1,1]
    ax4.semilogy(vnirv, vnirk, label='VNIR k')
    ax4.semilogy(even_vnirv, even_vnirk, '--r', label='Evenly spaced VNIR k')
    ax4.semilogy(new_vnirv, new_vnirk, '-.g', label = 'VNIR k without noisy end')
    ax4.legend()
    ax4.set_ylabel('k')
    ax4.set_xlabel('Wavenumber(cm e-1)')
    ax4.set_title('VNIR k')
    ax4.invert_xaxis()

    #Make sure we are combining this correctly
    #We want to be in descending wavenumbers
    #Find the low end of vnirv (high wavelength)
    if new_vnirv[0] > new_vnirv[-1]:
        vnirv_end = new_vnirv[-1]
    else:
        new_vnirv = np.flip(new_vnirv, axis=0)
        new_vnirk = np.flip(new_vnirk, axis=0)
        vnirv_end = new_vnirv[-1]

    #Find High end of v 
    if v[0] > v[-1]:
        v_end = v[-1]
    else:
        v = np.flip(v, axis=0)
        k = np.flip(k, axis=0)
        v_end = v[0] # Note that v[-1] and v_end will not be the same

    plt_data = [
                ['MIR k', v, k],
                ['VNIR k', lam, vnirk],
                ['Wavenumber(cm e-1) vs K', vnirv, vnirk],
                ['Evenly Spaced VNIR k', even_vnirv, even_vnirk],
                ['VNIR k without noisy end', new_vnirv, new_vnirk]
               ]

    cache = (v, k, lam, vnirk, vnirv, rev_vnirk, rev_vnirv, even_vnirk, even_vnirv, new_vnirk, new_vnirv, vnirv_end, v_end, vdiff, vnirvdiff)

    if adjType == '0':
        pltdata2, kset = adj_method_0(cache)
    elif adjType == '1':
        pltdata2, kset = adj_method_1(cache)
    elif adjType == '2':
        pltdata2, kset = adj_method_2(cache)
    elif adjType == '3':
        pltdata2, kset = adj_method_3(cache)

    plt_data.extend(pltdata2)

    return plt_data, kset

def adj_method_0(cache):
    plt_data = []
    (v, k, lam, vnirk, vnirv, rev_vnirk, rev_vnirv, even_vnirk, even_vnirv, new_vnirk, new_vnirv, vnirv_end, v_end, vdiff, vnirvdiff) = cache

    if v_end < vnirv_end:
        vcon = np.linspace(vnirv_end-vdiff, v_end+vdiff, round(abs((v_end+vdiff)-(vnirv_end-vdiff))/vdiff));
        m = (k[0]-new_vnirk[-1])/(v[0]-new_vnirv[-1]);
        #remember that they are arranged by decreasing wavenumber
        kcon = np.zeros(vcon.shape);
        for i in range(len(vcon)):
            kcon[i] = m*(vcon[i]-new_vnirv[-1])+new_vnirk[-1]
        fullv = np.concatenate((new_vnirv, vcon, v), axis=None)
        fullk = np.concatenate((new_vnirk, kcon, k), axis=None)

        fig1, ax1 = plt.subplots(figsize=(6, 4), frameon=False)
        ax1.semilogy(v, k, label='MIR k')
        ax1.semilogy(new_vnirv, new_vnirk, label='Evenly spaced VNIR k')
        ax1.semilogy(vcon, kcon,  label = 'Extended k')
        ax1.semilogy(fullv, fullk,'-', label = 'Whole k')
        ax1.legend()
        ax1.set_xlabel('Wavenumber(cm e-1)')
        ax1.set_ylabel('k')
        ax1.set_title('Combined k')
        ax1.invert_xaxis()
        plt_data.extend([['Extended K', vcon, kcon],['Whole k', fullv, fullk]])

        nv = v
        nk = k
    elif v_end >= vnirv_end:
        #cut some off MIR to connect the dots
        min_index = np.argmin(abs(v-vnirv_end))
        v_end = v[min_index]
        nv = v[min_index+1:]
        nk = k[min_index+1:]
        #remember that they are arranged by decreasing wavenumber
        fullv = np.concatenate((new_vnirv,nv), axis=None)
        fullk = np.concatenate((new_vnirk,nk), axis=None)
        
        fig1, ax1 = plt.subplots(figsize=(6, 4), frameon=False)
        ax1.semilogy(v, k, label='MIR k')
        ax1.semilogy(nv, nk, label='Cropped MIR k')
        ax1.semilogy(new_vnirv, new_vnirk, label = 'Evenly spaced VNIR k')
        ax1.semilogy(fullv, fullk,'-', label = 'Whole k')
        ax1.legend()
        ax1.set_xlabel('Wavenumber(cm e-1)')
        ax1.set_ylabel('k')
        ax1.set_title('Combined k')
        ax1.invert_xaxis()
        
        plt_data.extend([['Cropped MIR K', nv, nk],['Whole k', fullv, fullk]])

    kset = (vnirv, vnirk, fullv, fullk)

    return plt_data, kset

def adj_method_1(cache):
    plt_data = []
    (v, k, lam, vnirk, vnirv, rev_vnirk, rev_vnirv, even_vnirk, even_vnirv, new_vnirk, new_vnirv, vnirv_end, v_end, vdiff, vnirvdiff) = cache

    #Make a linearly increasing array
    y = np.linspace(0.0000001, 1, len(k))
    #Make an array that is exponential in logspace
    y_scale = 10**(y**10) # Eli thinks this can be a user input -- but not in 2019....
    #Flip it so that it is multiplying correctly
    y_scale = np.flip(y_scale, axis=0)
    #Adjust MIR k by dividing the scaling factor
    nk = k / y_scale

    #Plot
    fig1, ax1 = plt.subplots(figsize=(6, 4), frameon=False)
    ax1.semilogy(v, k, label='MIR k')
    ax1.semilogy(v, nk, label='Scaled MIR k')
    ax1.semilogy(even_vnirv, even_vnirk,  label = 'Evenly spaced VNIR k')
    ax1.legend()
    ax1.set_xlabel('Wavenumber(cm e-1)')
    ax1.set_ylabel('k')
    ax1.set_title('VNIR k')
    ax1.invert_xaxis()
    plt_data.append(['Evenly Spaced MIR K', v, nk])

    #Now quick and dirty linear connection
    if v_end < vnirv_end:
        vcon = np.linspace(vnirv_end - vdiff, vnirv_end + vdiff, round(abs((v_end + vdiff) - (vnirv_end - vdiff)) / vdiff))
        m = (nk[0] - new_vnirk[-1]) / ( v[0] - new_vnirv[-1])
        # Remember that they are arranged by decreasing wavenumber
        kcon = np.zeros(vcon.shape)
        for i in range(len(vcon)):
            kcon[i] = m * ( vcon[i] - new_new_vnirv[-1]) + new_vnirk[-1]
        full_v = np.concatenate((new_vnirv, vcon, v), axis=None) # Default axis = 0. Row-wise
        full_k = np.concatenate((new_vnirk, kcon, nk), axis=None)

        fig2, ax2 = plt.subplots(figsize=(6, 4), frameon=False)
        ax2.semilogy(v, k, label='MIR k')
        ax2.semilogy(v, nk, label='Scaled MIR k')
        ax2.semilogy(new_vnirv, new_vnirk, label = 'Evenly spaced VNIR k')
        ax2.semilogy(vcon, kcon, label = 'Extended k')
        ax2.semilogy(full_v, full_k,'-', label = 'Whole k')
        ax2.legend()
        ax2.set_xlabel('Wavenumber(cm e-1)')
        ax2.set_ylabel('k')
        ax2.set_title('VNIR k')
        ax2.invert_xaxis()
        plt_data.extend([['Scaled MIR K', v, nk],['Whole k', fullv, fullk], ['Extended k', vcon, kcon]])

    elif v_end >= vnirv_end:
        #Cut some off MIR to connect the dots
        min_index = np.argmin(abs(v - vnirv_end))
        v_end = v[min_index]
        nv = v[(min_index + 1):]
        nk = nk[(min_index+1):]

        #Remember that they are arranged by decreasing wavenum
        fullv = np.concatenate((new_vnirv, nv), axis=None) # Default axis = 0. Row-wise
        fullk = np.concatenate((new_vnirk, nk), axis=None)

        fig2, ax2 = plt.subplots(figsize=(6, 4), frameon=False)
        ax2.semilogy(v, k, label='MIR k')
        ax2.semilogy(nv, nk, label='scaled MIR k')
        ax2.semilogy(new_vnirv, new_vnirk, label = 'Evenly spaced VNIR k')
        ax2.semilogy(fullv, fullk,'-', label = 'Whole k') ## There is a additional label in the matlab code
        ax2.legend()
        ax2.set_xlabel('Wavenumber(cm e-1)')
        ax2.set_ylabel('k')
        ax2.set_title('Combined k')
        ax2.invert_xaxis()
        plt_data.extend([['Scaled MIR K', nv, nk],['Whole k', fullv, fullk]])
    
    kset = (vnirv, vnirk, fullv, fullk)

    return plt_data, kset

def adj_method_2(cache):
    plt_data = []
    (v, k, lam, vnirk, vnirv, rev_vnirk, rev_vnirv, even_vnirk, even_vnirv, new_vnirk, new_vnirv, vnirv_end, v_end, vdiff, vnirvdiff) = cache
    if v_end < vnirv_end:
        new_end = v_end # - vnirvdiff -- taking this out leads to a double point later
        #Fit the end of the data so that it will meet the MIR data
        ## Requires different functions sometimes, try POLY3, POLY4
        # deifine end of curve to fit
        #cut off any noise at end commented out 7/17/18 FEEL FREE TO CHANGE
        #lam=lam(1:noisend);
        #vnirk=vnirk(1:noisend);
        #set area to fit with a curve FEEL FREE TO CHANGE
        ep = len(new_vnirv)
        sp = ep - 30
        short_vnirv = new_vnirv[sp:ep]
        short_vnirk = new_vnirk[sp:ep]

        #Fit Curve
        fcoef = np.polyfit(short_vnirv, short_vnirk, 1)

        #extrapolate the end of lam
        #find spacing
        vspace = round(abs(new_vnirv[-1] - new_end) / vnirvdiff)
        vnirvext = np.linspace(new_vnirv[-1], new_end, vspace)
        #evaluate the function over the new range
        extrak=evalPoly(fcoef, vnirvext)

        #plot it to see how we did
        fig1, ax1 = plt.subplots(figsize=(6, 4), frameon=False)
        ax1.semilogy(vnirv, vnirk, label='VNIR k')
        ax1.semilogy(new_vnirv, new_vnirk,  label='Shortened VNIR k')
        ax1.semilogy(even_vnirv, even_vnirk,label = 'Evenly spaced VNIR k')
        ax1.semilogy(vnirvext, extrak, label = 'Extended k')
        ax1.legend()
        ax1.set_xlabel('Wavenumber(cm e-1)')
        ax1.set_ylabel('k')
        ax1.set_title('Extended VNIR k')
        ax1.invert_xaxis()
        plt_data.append(['Extended k', vnirvext, extrak])
        
        #combine fit end with data (for this step only)
        fvnirk=np.concatenate((new_vnirk,extrak), axis=None)
        fv=np.concatenate((new_vnirv,vnirvext), axis=None)
    else:
        fvnirk=new_vnirk # oveflap dealt with in next section for this method
        fv=new_vnirv; 

    #HERE IS WHERE WE ADJUST THE MIR DATA DOWN TO THE VNIR DATA (n would go up
    #based on epsilon infinity values so k goes down)
    #deal with potential overlap
    if max(v) > min(fv):
        minloc = np.argmin(abs(v-min(fv)))
        nv=v[minloc+1:]
        nk=k[minloc+1:]
    else:#no overlap
        nv=v
        nk=k

    #remember we know we are in descending frequency space
    #find difference between the two k values
    offset=nk[0]-fvnirk[-1]
    #DEALING WITH THE POSSIBILITY THAT LOWERING MIRK MAKES NEG
    #print error message
    if min(nk-offset) <= 0:
        raise ValueError('MIR k will be negative, choose another adj_method')
    elif min(nk-offset) > 0:
        adjk = nk-offset
        
        #plot it to make sure it worked
        fig2, ax2 = plt.subplots(figsize=(6, 4), frameon=False)
        ax2.semilogy(v, k, label='MIR k')
        ax2.semilogy(nv, adjk, label='Adjusted MIR k')
        ax2.legend()
        ax2.set_xlabel('Wavenumber(cm e-1)')
        ax2.set_ylabel('k')
        ax2.set_title('MIR k Adjustment')
        ax2.invert_xaxis()
        plt_data.append(['Adjusted MIR k', nv, adjk])

        fullv = np.concatenate((fv,nv), axis=None)
        fullk = np.concatenate((fvnirk, adjk), axis=None)
    
        fig3, ax3 = plt.subplots(figsize=(6, 4), frameon=False)
        ax3.semilogy(nv, adjk, label='Adjusted MIR k')
        ax3.semilogy(fv, fvnirk, label='VNIR k')
        ax3.semilogy(fullv, fullk, '-', label='Combined k')
        ax3.legend()
        ax3.set_xlabel('Wavenumber(cm e-1)')
        ax3.set_ylabel('k')
        ax3.set_title('Combined k')
        ax3.invert_xaxis()
        plt_data.extend([['Combined k', fullv, fullk], ['VNIR k', fv, fvnirk]])
    
    kset = (vnirv, vnirk, fullv, fullk)

    return plt_data, kset

def adj_method_3(cache): #this will bring MIRk down some VNIRk up some and draw line
    plt_data = []
    (v, k, lam, vnirk, vnirv, rev_vnirk, rev_vnirv, even_vnirk, even_vnirv, new_vnirk, new_vnirv, vnirv_end, v_end, vdiff, vnirvdiff) = cache

    if v_end < vnirv_end:
        high = min(k)
        low = max(new_vnirk)
        offset = high - low
        adjk = k - offset
        vcon = np.linspace(vnirv_end-vdiff,v_end+vdiff, round(abs((v_end+vdiff)-(vnirv_end-vdiff))/vdiff))
        m = (adjk[0]-new_vnirk[-1])/(v[0]-new_vnirv[-1])
        #remember that they are arranged by decreasing wavenumber
        kcon = np.zeros(vcon.shape)
        for i in range(len(vcon)):
            kcon[i] = m*(vcon[i]-new_vnirv[-1])+new_vnirk[-1]

        fullv = np.concatenate((new_vnirv,vcon,v), axis=None)
        fullk = np.concatenate((new_vnirk,kcon,adjk), axis=None)

        fig1, ax1 = plt.subplots(figsize=(6, 4), frameon=False)
        ax1.semilogy(v, k, label='MIR k')
        ax1.semilogy(v, adjk, label='Adjusted MIR k')
        ax1.semilogy(new_vnirv, new_vnirk, label='Evenly Spaced VNIR k')        
        ax1.semilogy(vcon, kcon, label='Extended k')        
        ax1.semilogy(fullv, fullk,'-',label='Whole k')
        ax1.legend()
        ax1.set_xlabel('Wavenumber(cm e-1)')
        ax1.set_ylabel('k')
        ax1.set_title('Combined k')
        ax1.invert_xaxis()
        plt_data.extend([['Adjusted MIR K', v, adjk],['Whole k', fullv, fullk], ['Extended k', vcon, kcon]])

        nv = v
        nk = k
    elif v_end >= vnirv_end:
        high = min(k)
        low = max(new_vnirk)
        offset = high-low
        adjk = k - offset
        #cut some off MIR to connect the dots
        min_index = np.argmin(abs(v-vnirv_end))
        v_end = v[min_index]
        nv = v[min_index+1:]
        nk = k[min_index+1:]
        #remember that they are arranged by decreasing wavenumber
        fullv = np.concatenate((new_vnirv,nv), axis=None)
        fullk = np.concatenate((new_vnirk,nk), axis=None)
        
        fig1, ax1 = plt.subplots(figsize=(6, 4), frameon=False)
        ax1.semilogy(v, k, label='MIR k')
        ax1.semilogy(v, adjk, label='Adjusted MIR k')        
        ax1.semilogy(nv, nk, label='Cropped MIR k')    
        ax1.semilogy(new_vnirv, new_vnirk, label='Evenly Spaced VNIR k')
        ax1.semilogy(fullv, fullk,'-', label='Whole k')
        ax1.legend()
        ax1.set_xlabel('Wavenumber(cm e-1)')
        ax1.set_ylabel('k')
        ax1.set_title('Combined k')
        ax1.invert_xaxis()
        plt_data.extend([['Adjusted MIR K', v, adjk],['Whole k', fullv, fullk], ['Cropped MIR k', nv, nk]])
        
    kset = (vnirv, vnirk, fullv, fullk)

    return plt_data, kset

#starting with highest power first
def evalPoly(lst, x):
    lst = lst[::-1] ## Reverse to lowest power the way we evaluate
    total = []
    for power, coeff in enumerate(lst):
        total.append((x**power) * coeff)
    return sum(total)

def no_mir_data(lam, vnirk):
    vnirv = 10000 / lam
    fig, axes = plt.subplots(figsize=(12,3), nrows=1, ncols=2)
    
    #Plot vnirk vs wavelength
    #If you are changing anything here please change in MaskerKCombine() where the same is plotted    
    ax1 = axes[0]
    ax1.plot(lam, vnirk)
    ax1.set_ylabel('k')
    ax1.set_xlabel('Wavelength(um)')
    ax1.set_title('VNIR k')
    
    # get a frequency space vector for vnirk
    vnirv = 10000 / lam

    #If you are changing anything here please change in MaskerKCombine() where the same is plotted  
    ax2 = axes[1]
    ax2.semilogy(vnirv, vnirk)
    ax2.set_ylabel('k')
    ax2.set_xlabel('Wavenumber(cm e-1)')
    ax2.set_title('Wavenumber(cm e-1) vs K')
    ax2.invert_xaxis() 

    plt_data = [
                ['VNIR k', lam, vnirk],
                ['Wavenumber(cm e-1) vs K', vnirv, vnirk]
               ]

    kset = vnirv, vnirk, None, None
    
    return plt_data, kset

def MasterSSKK(kset, anchor, iter, wavelength, n1, lstart, lend):
    #I am using the VNIRV from the previous step - unlike the MATLAB code - Check with ELI
    vnirv, vnirk, fullv, fullk = kset
    lam = wavelength
    
    #get info on first and last element to re interpolate
    first = vnirv[0]
    last = vnirv[-1]
    sizev = len(vnirv)

    ## Skip -- kcombine option added 
    if fullv is None and fullk is None:

        #Check if vnirv, vnirk is in ascending
        if vnirv[-1] < vnirv[0]:
            vnirv = np.flip(vnirv, axis=0)
            vnirk = np.flip(vnirk, axis=0)

        fullv = np.linspace(first, last, sizev)
        interp_ob = interpolate.interp1d(vnirv, vnirk, kind='linear')
        fullk = interp_ob(fullv)

    kap = fullk
    v = fullv


    #Check if fullv, fullk is in ascending
    if v[-1] < v[0]:
        v = np.flip(v, axis=0)
        kap = np.flip(kap, axis=0)

    #Size of data set
    sizev = len(v)

    #define the real index of refraction and the anchor point
    lam1 = float(anchor) #average of sodium D doublet
    v1 = 10000/lam1 #converted to frequency

    # intnb : specifies the number of intervals to run approximation over
    # (originally 1000, but that seemed excessive)
    intnb=1 #10

    #make start point and end point values for integration that place k at the
    #center of each range
    dv = v[1] - v[0]
    dx = dv / intnb #spacing for all x's
    halfv = dv/2;
    offset = np.linspace(-halfv, halfv, intnb) #10 evenly spaced steps in every dv
    xx = np.matlib.repmat(v, intnb, 1) + np.matlib.repmat(offset, sizev, 1).T
    #xx is ten rows to comput simultaneously, 1 for     each offset for each v

    # compute all the bits that don't change in the loop.
    v_sq = v**2
    v1_sq = v1**2
    n = (2/math.pi)*(v_sq-v1_sq)*dx #rename. n is confusing
    xx_sq = xx**2 #v'^2
    tmp = xx_sq - v1_sq #v'^2-v1^2
    numerator = xx * np.matlib.repmat(kap, intnb, 1) #v'*k(v')

    for j in range(len(v)):
        # progress counter
        if j % 100 == 0:
            print('Iteration '+str(j)+' of '+str(len(v)))
        #Compute function over all xx for each v_j
        yy = numerator /((xx_sq-v_sq[j]) * tmp)
        # calculate the real index of refraction with integral over intnb grid
        n[j] = n[j] * np.sum(np.trapz(yy))
    n += n1

    #extract VNIR n and save to different file for later
    vlam = 10000 /v 
    nlam = n

    #make sure in same order as k will be
    if vlam[0] > vlam[-1]:
        vlam = np.flip(vlam, axis=0)
        nlam = np.flip(n, axis=0)
 
    lamdiff = lam[1] - lam[0]
    # lstart comes from UV

    #get actual k wavelength becuase stupid floating points are not working
    ### what is wavelength
    ksize=int(len(wavelength))
    vislam=wavelength[:ksize]

    #lstart=lstart+iter*lamdiff;
    dist = abs(vlam - lend)
    minloc = np.argmin(dist)
    upend = minloc
    dist = abs(vlam - lstart)
    minloc = np.argmin(dist)
    downend = minloc

    ## Python finds a different minimum than MATLAB does by one point less in the array in Python

    visvlam = vlam[downend:upend]
    vnirn = nlam[downend:upend]
    #extrapolate to k spacing
    interp_ob  = interpolate.interp1d(visvlam,  vnirn, fill_value='extrapolate') ## Need to check if this extrapolates
    visn = interp_ob(vislam)
    
    fig1, ax1 = plt.subplots(figsize=(6, 4), frameon=False)
    ax1.plot(visvlam, vnirn, label = 'n')
    ax1.plot(vislam, visn, label = 'reinterpolated n')
    ax1.legend()
    ax1.set_xlabel('Wavelength(um)')
    ax1.set_ylabel('n')
    ax1.set_title('VNIR n')
    
    ## Note: We removed lines that save the lstart and lend -- 
    ## which are the end points that may or may not have been removed

    plt_data = [
                ['n', visvlam, vnirn],
                ['Reinterpolated n', vislam, visn]
               ]

    vars = lstart, lend, lamdiff, vislam, visn

    return plt_data, vars

def solve_phase(phase_files, params):
    
    (lstart2, lend2, low, UV, lamdiff, minScale, maxScale, minOffset, maxOffset, maxfun, funtol, xtol, spts, diff_step
     vislam, visn, wavelength, k, fit_order, phaseAngleCount, phaseGrainList, phase_bcsd, ffs, hapke) = params

    phase_file_key_list = range(phaseAngleCount)
    grain_samples = len(phaseGrainList.keys())
    nfiles = phaseAngleCount * grain_samples
    prow, pcol = phaseGrainList[0][0].data.shape
    
    #wave is not processed, wavelength is from pp_spectra
    wave = phaseGrainList[1][0].data[:,0] #Shape (N,2) -- Take only first column

    #find indices of those values in the wavelength vector
    #recreate the wavelength vector
    lowind = np.argmin(abs(wave-low))  
    highind = np.argmin(abs(wave-lend2)) + 1 #because python is not inclusive of the arrays
    wave = wave[lowind:highind]
    prow2 = wave.shape[0]

    #extract reflectance data over new, smaller range and make it just data
    full_phase = np.zeros((grain_samples, phaseAngleCount, prow, pcol))
    for g in range(grain_samples):
        for i, key in enumerate(phaseGrainList[g]):
            full_phase[g, i, :, :] = phaseGrainList[g][i].data

    #This first reshapes full_phase to g1p1,g1p2,g1p3,g2p1....gnpn....
    #Next it selects the second column from the phase file
    #Next it cuts off the rows from lowind to highind
    #once this is done phase data is now of shape nfiles,prow2,1
    phasedata = np.reshape(full_phase, (nfiles, prow, pcol))[:,:,1][lowind:highind] #phasedata = nfiles, prow2

    #extend data into UV based on assumptions
    #extrapolate left side of k to 0.2 using ahrekiel's constant method
    #Cloutis 2008b shows that jarsotie is essentially flat from 200-400nm

    leftw = np.arange(UV, wave[0], lamdiff)
    head = len(leftw)

    #fit the end of the data 
    #REQUIRES DIFFERENT FUCNTION SOMETIMES, TRY 'poly3' and 'poly4'
    #'linearinterp' works better than 'poly1'
    #type help curvefit or doc fit to bring up more info
    #define end of curve to fit (must be column vectors)
    #upper end of fittable range or feature
    sp = 0
    #change ep to end of useable feature (index not wavelength)
    ep = sp+5;
    shortlam = wave[sp:ep]
    prow3 = shortlam.shape[0]
    feature = phasedata[:, sp:ep] #feature = nfiles, prow3
     
    #Fit Curve
    fit_coefs_count = fit_order + 1 
    #polyfit returns highest power first
    fcoef = np.polyfit(shortlam, feature.T, fit_order).T #fcoef = nfiles, fit_coefs_count

    UVdata = np.zeros((nfiles, head))
    for i in range(nfiles):
        UVdata[i,:] = evalPoly(fcoef[i], leftw)
       
    longphasedata = np.concatenate((UVdata, phasedata[:,:]),axis=1) #longphasedata = nfiles, sizep
    wave = np.concatenate((leftw, wave[:]),axis=0)

    v = vislam
    n = visn

    #make sure the arrays are all the same length
    # If n wavelength array is shorter than k wavelength array
    # Then we cut k
    kwave = wavelength
    if v[0] > kwave[0]:
        offset = v[0] - kwave[0]
        indexoff = round(offset/lamdiff)
        newstart = indexoff + 1
        kwave = kwave[newstart:]
        k = k[newstart:]
    #fix start point
    # If data wavelength array is longer than k wavelength array
    # Then we cutting data wavelength
    if wave[0] < kwave[0]:
        offset = kwave[0] - wave[0]
        indexoff = round(offset/lamdiff)
        newstart = indexoff + 1
        wave = wave[newstart:]
        v = v[newstart:]
        n = n[newstart:]
        longphasedata = longphasedata[:,newstart:]

    #fix end point
    if kwave[-1] < wave[-1]:
        newend = len(kwave) # This is inclusive -- Eli and Aish did a small test together
        wave = wave[:newend]
        v = v[:newend]
        n = n[:newend]
        longphasedata = longphasedata[:, :newend]
    
    sizep=len(wave) #225
    
    #-------------------------------------------
    #we don't like to play favorites but...
    favn = np.repeat(n[np.newaxis,:], nfiles, axis=0) #This is n repeated nfiles, sizep
    favk = np.repeat(k[np.newaxis,:], nfiles, axis=0) #This is n repeated totalspectra, sizep
    fav_wave = np.repeat(wave[np.newaxis,:], nfiles, axis=0) #This is n repeated totalspectra, sizep
    
    #build s and d which are of shape nfiles, 1
    #build b and c which are sizep*gs, 1
    scalar_b_list = []
    scalar_c_list = []
    scalar_s_list = []
    scalar_d_list = []
    scalar_lb_b_list = []
    scalar_lb_c_list = []
    scalar_lb_s_list = []
    scalar_lb_d_list = []
    scalar_ub_b_list = []
    scalar_ub_c_list = []
    scalar_ub_s_list = []
    scalar_ub_d_list = []

    scalar_ff_list = []

    if hapke.needs_bg:
        scalar_b0_list = []
        scalar_lb_b0_list = []
        scalar_ub_b0_list = []
        
        scalar_h_list = []
        scalar_lb_h_list = []
        scalar_ub_h_list = []

    for i in range(grain_samples):
        bi = phase_bcsd[i][0][0] ## each is a tuple of guess, lb, ub and each within is b,c,s,d
        ci = phase_bcsd[i][0][1] 
        si = phase_bcsd[i][0][2] 
        di = phase_bcsd[i][0][3]
        
        lb_bi = phase_bcsd[i][1][0] ## each is a tuple of guess, lb, ub and each within is b,c,s,d
        lb_ci = phase_bcsd[i][1][1] 
        lb_si = phase_bcsd[i][1][2] 
        lb_di = phase_bcsd[i][1][3]
        
        ub_bi = phase_bcsd[i][2][0] ## each is a tuple of guess, lb, ub and each within is b,c,s,d
        ub_ci = phase_bcsd[i][2][1] 
        ub_si = phase_bcsd[i][2][2] 
        ub_di = phase_bcsd[i][2][3]

        if hapke.needs_bg:
            b0i = phase_bcsd[i][0][4]
            hi = phase_bcsd[i][0][5]

            lb_b0i = phase_bcsd[i][1][4]
            lb_hi = phase_bcsd[i][1][5]

            ub_b0i = phase_bcsd[i][2][4]
            ub_hi = phase_bcsd[i][2][5]

        scalar_b_list.append(np.repeat(bi, sizep, axis=0))
        scalar_c_list.append(np.repeat(ci, sizep, axis=0))
        scalar_s_list.append(si) #np.repeat(si, phaseAngleCount, axis=0)
        scalar_d_list.append(di) #np.repeat(di, phaseAngleCount, axis=0)
        
        scalar_ff_list.append(ffs[i]) #np.repeat(ffs[i], phaseAngleCount, axis=0)

        scalar_lb_b_list.append(np.repeat(lb_bi, sizep, axis=0))
        scalar_lb_c_list.append(np.repeat(lb_ci, sizep, axis=0))
        scalar_lb_s_list.append(lb_si) #np.repeat(lb_si, phaseAngleCount, axis=0)
        scalar_lb_d_list.append(lb_di) #np.repeat(lb_di, phaseAngleCount, axis=0)
         
        scalar_ub_b_list.append(np.repeat(ub_bi, sizep, axis=0))
        scalar_ub_c_list.append(np.repeat(ub_ci, sizep, axis=0))
        scalar_ub_s_list.append(ub_si) #np.repeat(ub_si, phaseAngleCount, axis=0)
        scalar_ub_d_list.append(ub_di) #np.repeat(ub_di, phaseAngleCount, axis=0)

        if hapke.needs_bg:
            scalar_b0_list.append(b0i)
            scalar_h_list.append(hi)
            scalar_lb_b0_list.append(lb_b0i) #np.repeat(lb_si, phaseAngleCount, axis=0)
            scalar_lb_h_list.append(lb_hi) #np.repeat(lb_di, phaseAngleCount, axis=0)
            scalar_ub_b0_list.append(ub_b0i) #np.repeat(ub_si, phaseAngleCount, axis=0)
            scalar_ub_h_list.append(ub_hi) #np.repeat(ub_di, phaseAngleCount, axis=0)
                         
    b = np.hstack(tuple(scalar_b_list))
    c = np.hstack(tuple(scalar_c_list))
    s = np.hstack(tuple(scalar_s_list)) #grain_samples long
    d = np.hstack(tuple(scalar_d_list)) #grain_samples long
    ff = np.hstack(tuple(scalar_ff_list))

    lb_b = np.hstack(tuple(scalar_lb_b_list))
    lb_c = np.hstack(tuple(scalar_lb_c_list))
    lb_s = np.hstack(tuple(scalar_lb_s_list)) #grain_samples long
    lb_d = np.hstack(tuple(scalar_lb_d_list)) #grain_samples long

    ub_b = np.hstack(tuple(scalar_ub_b_list))
    ub_c = np.hstack(tuple(scalar_ub_c_list))
    ub_s = np.hstack(tuple(scalar_ub_s_list)) #grain_samples long
    ub_d = np.hstack(tuple(scalar_ub_d_list)) #grain_samples long

    if hapke.needs_bg:
        b0 = np.hstack(tuple(scalar_b0_list)) #grain_samples long
        h = np.hstack(tuple(scalar_h_list)) #grain_samples long 
        lb_b0 = np.hstack(tuple(scalar_lb_b0_list)) #grain_samples long
        lb_h = np.hstack(tuple(scalar_lb_h_list)) #grain_samples long
        ub_b0 = np.hstack(tuple(scalar_ub_b0_list)) #grain_samples long
        ub_h = np.hstack(tuple(scalar_ub_h_list)) #grain_samples long

        #stack b,c,s,d
        bcsd_guess = np.hstack((b,c,s,d, b0, h)) #b,c == sizep*gs, s,d, b0, h = grain samples
        bcsd_lb = np.hstack((lb_b,lb_c,lb_s,lb_d, lb_b0, lb_h)) #b,c == sizep*gs, s,d, b0, h = grain samples
        bcsd_ub = np.hstack((ub_b,ub_c,ub_s,ub_d, ub_b0, ub_h)) #b,c == sizep*gs, s,d, b0, h = grain samples
    else:
        #stack b,c,s,d
        bcsd_guess = np.hstack((b,c,s,d)) #b,c == sizep*gs, s,d = grain samples
        bcsd_lb = np.hstack((lb_b,lb_c,lb_s,lb_d)) #b,c == sizep*gs, s,d = grain samples
        bcsd_ub = np.hstack((ub_b,ub_c,ub_s,ub_d)) #b,c == sizep*gs, s,d = grain samples
    
    bcsd_guess = np.append([1,0], bcsd_guess) # scale and offset guesses
    bcsd_lb = np.append([minScale,minOffset], bcsd_lb) #bounds are inclusive
    bcsd_ub = np.append([maxScale,maxOffset], bcsd_ub)

    #thetai and thetae should be nfiles, 1
    thetai_list = []
    thetae_list = []
    for g in range(grain_samples):
        for i, key in enumerate(phaseGrainList[g]):
            thetai_list.append(phaseGrainList[g][i].incident_angle)
            thetae_list.append(phaseGrainList[g][i].emission_angle)

    thetai = np.asarray(thetai_list) [:, np.newaxis]
    thetae = np.asarray(thetae_list) [:, np.newaxis]

    #I am using the vector copy - the scalar copy is with progstate safely
    #For Bg: init angles takes care of resetting g for each thetai and thetae
    hapke._init_angles(thetai, thetae)
    # scat efficiency takes care of init refraction

    def obj_fun(coefp):
        loss = 0
        rc, scale, offset = phase_rc(coefp, hapke, sizep, grain_samples, phaseAngleCount, favk, fav_wave, favn, ff)
        loss += ((rc - longphasedata)**2).sum()
        return np.sqrt(loss)
    
    
    start_points = np.empty((spts, len(bcsd_guess)))
    start_points[0] = bcsd_guess

    #Initialize random start points
    for i in range(1, spts):
        start_points[i] = np.random.uniform(bcsd_lb, bcsd_ub)

    bounds = np.row_stack((bcsd_lb, bcsd_ub))
    solutions = []
    #For each start point - minimize the least square error and append it to solutions.

    valcnt = 6 if hapke.needs_bg else 4
    _, axes = plt.subplots(figsize=(17,grain_samples*3), nrows=grain_samples, ncols=valcnt)
    fig1, ax1 = plt.subplots(figsize=(6,4), frameon=False)

    axes[0,0].set_title('Converged b')
    axes[0,1].set_title('Converged c')
    axes[0,2].set_title('s')
    axes[0,3].set_title('D')

    if hapke.needs_bg:
        axes[0,4].set_title('b0')
        axes[0,5].set_title('h')
    
    s_st = {g: [] for g in range(grain_samples)} 
    s_ep = {g: [] for g in range(grain_samples)}
    d_st = {g: [] for g in range(grain_samples)}
    d_ep = {g: [] for g in range(grain_samples)}

    if hapke.needs_bg:
        b0_st = {g: [] for g in range(grain_samples)} 
        b0_ep = {g: [] for g in range(grain_samples)}
        h_st = {g: [] for g in range(grain_samples)}
        h_ep = {g: [] for g in range(grain_samples)}

    plt_data = []

    for s, spt in enumerate(start_points):
        res = least_squares(obj_fun, spt, bounds=bounds, ftol = funtol, xtol = xtol, x_scale = 'jac', method = 'trf', max_nfev = maxfun, diff_step=diff_step, tr_solver='lsmr', verbose=2)
        solutions.append(res)

        #Time to build plots for each start point
        
        #Plotting the scaled and offset K
        spt_scale = res.x[0]
        spt_offset = res.x[1]
        #Take only k form the gigantic array of favk which is repeated nfiles times
        spt_k = favk[0] * spt_scale + spt_offset 
        ax1.semilogy(wave, spt_k, label='spt-'+str(s+1))
        plt_data.append(['offk_spt-'+str(s+1),wave, spt_k])
        ax1.set_title('Scaled and Offset K')
        ax1.set_xlabel('Wavelength')
        ax1.set_ylabel('K')
        ax1.legend() ## this is not good -- but easy to maintain

        ##Getting all the converged values
        con_sol = res.x[2:]
        b = con_sol[:sizep*grain_samples]
        c = con_sol[sizep*grain_samples:sizep*grain_samples*2]
        s_ep_all = con_sol[sizep*grain_samples*2:sizep*grain_samples*2+grain_samples]
        d_ep_all = con_sol[sizep*grain_samples*2+grain_samples:sizep*grain_samples*2+(2*grain_samples)]

        s_st_all = spt[sizep*grain_samples*2+2:sizep*grain_samples*2+grain_samples+2]
        d_st_all = spt[sizep*grain_samples*2+grain_samples+2:sizep*grain_samples*2+(2*grain_samples)+2]

        if hapke.needs_bg:
            b0_ep_all = con_sol[sizep*grain_samples*2+(2*grain_samples):sizep*grain_samples*2+(3*grain_samples)]
            h_ep_all = con_sol[sizep*grain_samples*2+(3*grain_samples):]

            b0_st_all = spt[sizep*grain_samples*2+(2*grain_samples)+2:sizep*grain_samples*2+(3*grain_samples)+2]
            h_st_all = spt[sizep*grain_samples*2+(3*grain_samples)+2:]

        ##Loading to the list previously
        for g in range(grain_samples):
            s_st[g].append(s_st_all[g])
            s_ep[g].append(s_ep_all[g])
            d_st[g].append(d_st_all[g])
            d_ep[g].append(d_ep_all[g])

            if hapke.needs_bg:
                b0_st[g].append(b0_st_all[g])
                b0_ep[g].append(b0_ep_all[g])
                h_st[g].append(h_st_all[g])
                h_ep[g].append(h_ep_all[g])

            bax = axes[g, 0]
            cax = axes[g, 1]
            sax = axes[g, 2]
            dax = axes[g, 3]

            if hapke.needs_bg:
                b0ax = axes[g, 4]
                hax = axes[g, 5]

            bax.plot(wave, b[len(wave)*g:len(wave)*(g+1)], label='sp:'+str(s+1))
            cax.plot(wave, c[len(wave)*g:len(wave)*(g+1)], label='sp:'+str(s+1))
            
            #Adding to Plot data
            plt_data.append(['b_sp-'+str(s+1)+'_gs-'+str(g+1), wave, b[len(wave)*g:len(wave)*(g+1)]])
            plt_data.append(['c_sp-'+str(s+1)+'_gs-'+str(g+1), wave, c[len(wave)*g:len(wave)*(g+1)]])
            
            ##Plot only after all the start points are run
            if s == (spts - 1):
                sax.plot(range(spts), s_st[g], '*r', range(spts), s_ep[g],'.g')
                dax.plot(range(spts), d_st[g], '*r', range(spts), d_ep[g],'.g')
                
                plt_data.append(['s_sp-'+str(s+1)+'_gs-'+str(g+1), range(spts), s_ep[g]])
                plt_data.append(['d_sp-'+str(s+1)+'_gs-'+str(g+1), range(spts), d_ep[g]])

                if hapke.needs_bg:                    
                    b0ax.plot(range(spts), b0_st[g], '*r', range(spts), b0_ep[g],'.g')
                    hax.plot(range(spts), h_st[g], '*r', range(spts), h_ep[g],'.g')
                    plt_data.append(['b0_sp-'+str(s+1)+'_gs-'+str(g+1), range(spts), b0_ep[g]])
                    plt_data.append(['h_sp-'+str(s+1)+'_gs-'+str(g+1), range(spts), h_ep[g]])

            bax.legend()
            cax.legend()
            bax.set_ylabel('file'+str(g+1))
            bax.set_xlabel('Wavelength')
            cax.set_xlabel('Wavelength')
            sax.set_xlabel('Start Points')
            dax.set_xlabel('Start Points')
            st = mpatches.Patch(color='red', label='Guess')
            ep = mpatches.Patch(color='green', label='Converged')
            sax.legend(handles=[st, ep])
            dax.legend(handles=[st, ep])
            if hapke.needs_bg:
                    b0ax.set_xlabel('Start Points')
                    hax.set_xlabel('Start Points')
                    b0ax.legend(handles=[st, ep])
                    hax.legend(handles=[st, ep])

    ## Plotting the best solution
    best_soln = min(solutions, key=lambda res: res.cost).x
    brc, bscale, boffset = phase_rc(best_soln, hapke, sizep, grain_samples, phaseAngleCount, favk, fav_wave, favn, ff)

    fig2, ax2 = plt.subplots(figsize=(8,6), frameon=False)
    for i in range(nfiles):
        ax2.plot(wave, longphasedata[i], '-b', label='LPD-'+str(i+1))
        ax2.plot(wave, brc[i], ':r', label='RC-'+str(i+1))
        plt_data.append(['RC-'+str(i+1), wave, brc[i]])
        plt_data.append(['LPD-'+str(i+1), wave, longphasedata[i]])
    ax2.set_xlabel('Wavelength')
    ax2.set_ylabel('Reflectance')
    ax2.set_title('Wavelength vs Reflectance')
    rc = mpatches.Patch(color='red', label='Model')
    lpd = mpatches.Patch(color='blue', label='Data')
    plt.legend(handles=[rc, lpd])

    return plt_data

def phase_rc(coefp, hapke, sizep, grain_samples, phaseAngleCount, favk, fav_wave, favn, ff):
        scale = coefp[0]
        offset = coefp[1]
        coefp = coefp[2:]
        b = coefp[:sizep*grain_samples]
        c = coefp[sizep*grain_samples:sizep*grain_samples*2]
        s = coefp[sizep*grain_samples*2:sizep*grain_samples*2+grain_samples]
        d = coefp[sizep*grain_samples*2+grain_samples:sizep*grain_samples*2+(2*grain_samples)]

        #repeating s and d - phase angle times and fitting the shape appropriately
        s = np.repeat(s, phaseAngleCount, axis=0)[:, np.newaxis]
        d = np.repeat(d, phaseAngleCount, axis=0)[:, np.newaxis]

        b0 = None
        h = None
        if hapke.needs_bg:
            b0 = coefp[sizep*grain_samples*2+(2*grain_samples):sizep*grain_samples*2+(3*grain_samples)]
            h = coefp[sizep*grain_samples*2+(3*grain_samples):]
            b0 = np.repeat(b0, phaseAngleCount, axis=0)[:, np.newaxis]
            h = np.repeat(h, phaseAngleCount, axis=0)[:, np.newaxis]

        #Now we are splittinng to their individual grain sizes 
        b_ustk = np.split(b, grain_samples)
        c_ustk = np.split(c, grain_samples)
        blst = []
        clst = []
        for i in range(grain_samples):
            blst.append(np.repeat(b_ustk[i][np.newaxis, :], phaseAngleCount, axis=0))
            clst.append(np.repeat(c_ustk[i][np.newaxis, :], phaseAngleCount, axis=0))
        allb = np.vstack(blst) #nfiles, sizep
        allc = np.vstack(clst) #nfiles, sizep

        favks = favk * scale + offset
        scat = hapke.scattering_efficiency(favks, fav_wave, d, s, favn)
        rc = hapke.radiance_coeff(scat, allb, allc, ff, b0, h)

        return rc, scale, offset
        