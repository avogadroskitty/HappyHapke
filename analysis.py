from __future__ import division, print_function
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
#Fix for Error in Section 3 - Solve For K in external Library Matplotlib
plt.switch_backend('agg')
from scipy.optimize import least_squares
from scipy.integrate import trapz


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
  return wavelength[idx1:idx2], spec[idx1:idx2]


def fit_left_side(wavelength, spec, UV, fit_order=0):
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
    idx = len(leftw) + 1 # Change 1 to optional user input
    fit_order = min(fit_order, len(leftw))
    # fit curves
    # Takes the wavelength, spectrum points as x,y and fits a polynomial on it. Degree is the fitorder variable 
    # poly1d makes a one-d polynomial from the co-efficients returned by the polyfit method 
    poly = np.poly1d(np.polyfit(wavelength[:idx], spec[:idx], fit_order))
    spec = np.concatenate((poly(leftw), spec))
  wavelength = np.concatenate((leftw, wavelength))
  return np.column_stack((wavelength, spec))


def preprocess_traj(traj, low, high, UV, fit_order=0):
  wave, spec = traj.T
  wave, spec = prepare_spectrum(wave, spec, low, high)
  return fit_left_side(wave, spec, UV, fit_order=fit_order)

#Solving for K - Logic -- setup the matrices here. For plotting get values from the hapke object - defined in hapke_model.py
def MasterHapke1_PP(hapke, traj, b, c, ff, s, D, debug_plots=False):
  wavelength, reflect = traj.T
  table_size = len(traj) * 2

  # make sure we have a scalar isow
  if hapke.needs_isow:
    assert np.isscalar(hapke.isow), 'MasterHapke1_PP requires scalar isow'

  # create table of increasing w (single scattering albedo) and use linear
  # interpolation to solve backwards from the real reflectance data
  w = np.linspace(0, 1, table_size, endpoint=False)
  rc = hapke.radiance_coeff(w, b, c, ff)
  w2 = np.interp(reflect, rc, w)

  # use the same trick to back-solve for k from w2, except we have to plug in
  # the (k/wavelength) term when interpolating
  # TODO: incorporate bounds on k to determine table bounds
  k_wave = np.logspace(-1, -7, table_size)
  scat = hapke.scattering_efficiency(k_wave, 1, D, s)
  k_wave2 = np.interp(w2, scat, k_wave)
  k = k_wave2 * wavelength

  if debug_plots:
    # calculate scattering efficiency for each solved k
    rc2 = hapke.radiance_coeff(w2, b, c, ff)
    ScatAlb = hapke.scattering_efficiency(k, wavelength, D, s)
    rc3 = hapke.radiance_coeff(ScatAlb, b, c, ff)

    #The _ is the figure and the axes object is stores in axes
    _, axes = plt.subplots(figsize=(10,4), nrows=2, ncols=2, sharex=True)
    # plot reflectance data and rc2 for comparison
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
def MasterHapke2_PP(hapke, spectra, coefg, lb, ub, ff, spts=1, **kwargs):
  """This program performs an iterative minimization using Hapke's radiative
  transfer theory to find a global and grain-size independent value of
  imaginary index of refraction, k."""
  no_of_grain_samples = len(spectra)
  total_guesses = no_of_grain_samples * 4 # 4 values (b,c,s,D) for each grain size

  wave = spectra['file2'][:,0]
  actuals = [spectra[key][:,1] for key in sorted(spectra.keys())]

  #Why 3 
  def obj_fn(coef):
    k = coef[total_guesses:]
    loss = 0
    for i, actual in enumerate(actuals):
      b, c, s, D = coef[i:total_guesses:no_of_grain_samples]
      #these are paired correctly now that it is sorted elsewhere
      scat = hapke.scattering_efficiency(k, wave, D, s)
      rc = hapke.radiance_coeff(scat, b, c, ff[i])
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
    res = least_squares(obj_fn, spt, bounds=bounds, ftol=1.0e-16, xtol=2.23e-16,x_scale = 'jac', method='trf', max_nfev=15000, **kwargs)
    solutions.append(res)
  return solutions


def optimize_global_k(hapke, spectra, coefg, lb, ub, ff, num_iters=5):
  '''Similar to MasterHapke2_PP, but uses a different approach.'''
  wave = spectra['file2'][:,0]
  actuals = [spectra[key][:,1] for key in ('file1', 'file2', 'file3')]
  bounds = np.row_stack((lb, ub))

  soln = coefg.copy()
  solutions = []
  for _ in range(num_iters):
    # step 1: optimize hapke parameters, holding k constant
    for i, actual in enumerate(actuals):
      idx = slice(i, 12, 3)
      res = _optimize_bcsD(hapke, wave, actual, soln[idx], bounds[:,idx],
                           ff[i], soln[12:])
      soln[idx] = res.x

    # step 2: optimize k, holding parameters constant
    res = _optimize_k(hapke, wave, actuals, soln[12:], bounds[:,12:],
                      soln[:12], ff)
    soln[12:] = res.x
    solutions.append(soln.copy())

  return solutions


def _optimize_bcsD(hapke, wave, actual, guess, bounds, ff, k):
  def obj_fn(coef):
    b, c, s, D = coef
    scat = hapke.scattering_efficiency(k, wave, D, s)
    rc = hapke.radiance_coeff(scat, b, c, ff)
    return np.linalg.norm(rc - actual)

  return least_squares(obj_fn, guess, bounds=bounds, verbose=1, max_nfev=1000)


def _optimize_k(hapke, wave, actuals, guess, bounds, coef, ff):
  def obj_fn(k):
    loss = 0
    for i, actual in enumerate(actuals):
      b, c, s, D = coef[i:12:3]
      scat = hapke.scattering_efficiency(k, wave, D, s)
      rc = hapke.radiance_coeff(scat, b, c, ff[i])
      loss += ((rc - actual)**2).sum()
    return np.sqrt(loss)

  return least_squares(obj_fn, guess, bounds=bounds, verbose=2, method='trf',
                       tr_solver='lsmr', max_nfev=20)


def MasterKcombine(mir_k_fname, mir_v_fname, wavelength, solved_k, fit_order=3,
                   resample=False):
  """Takes data from vnir k and dispersion k and turns it into a
  single adjusted vector with help from the user."""
  # load dispersion data
  k = loadmat_single(mir_k_fname).ravel()
  v = loadmat_single(mir_v_fname).ravel()

  lam = wavelength
  vnirk = solved_k

  # a little modeling magic to make the arrays meet
  # in case you cut off some noise at the end

  # get a frequency space vector for vnirk
  vnirv = 10000 / lam
  # find high end of v
  vend = v.max()
  # check to make sure we need to do this modelling
  if vend < vnirv.min():
    lamdiff = lam[1] - lam[0]
    newend = 10000/vend - lamdiff
    # fit the end of the data so that it will meet the MIR data
    poly = np.poly1d(np.polyfit(lam[-50:], vnirk[-50:], fit_order))
    # extrapolate the end of lam
    lamend = np.arange(lam[-1], newend, lamdiff)
    vnirk = np.concatenate((vnirk, poly(lamend)))
    lam = np.concatenate((lam, lamend))
    vnirv = 10000 / lam

  # HERE IS WHERE WE ADJUST THE MIR DATA DOWN TO THE VNIR DATA
  # (n would go up based on epsilon infinity values so k goes down)

  # make sure we are picking the low end of the VNIR data
  low = vnirk[-1] if lam[0] < lam[-1] else vnirk[0]
  # next make sure we are picking high frequency end of MIR data
  hi = k[0] if v[0] > v[-1] else k[-1]

  # correct for offset in MIR data
  offset = hi - low
  adjk = k - offset

  # make sure frequency is ascending
  if vnirv[0] > vnirv[-1]:
    vnirv = vnirv[::-1]
    vnirk = vnirk[::-1]
  if v[0] > v[-1]:
    v = v[::-1]
    adjk = adjk[::-1]

  # remove negative data
  mask = adjk > 0
  v = v[mask]
  adjk = adjk[mask]

  # cat them together (mid ir will always be first in frequency space)
  fullv = np.concatenate((v, vnirv))
  fullk = np.concatenate((adjk, vnirk))

  if resample:
    # reinterpolate so vector is evenly spaced
    evenfullv = np.linspace(fullv[0], fullv[-1], len(fullv))
    fullk = np.interp(evenfullv, fullv, fullk)
    fullv = evenfullv

  return np.column_stack((fullv, fullk))


def MasterSSKK(dispersion, n1, anchor, num_intervals=100):
  """This section determines the real index of refraction (n) from k,
  using a singly subtractive Kramers Kronig calculation."""
  # the result of MasterKcombine
  v, k = dispersion.T

  v1 = 10000 / anchor  # convert to frequency

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

  # TODO: look into vectorizing this fully. It's not too hard, but might be more
  # memory than we'd like to spend.
  for i, v_sq_i in enumerate(v_sq):
    # Compute function over all xx for each v_j
    yy = numerator / (xx_sq - v_sq_i)
    # calculate the real index of refraction with integral over the grid
    n[i] *= trapz(yy, axis=0).sum()
  n += n1

  return np.column_stack((v, n))


def MasterPhase1(hapke, spectra, coefg, lb, ub, ff):
  """Use data from multiple viewing geometries to calculate phase function
  parameters for a sample where k and n for thetai=30, thetae=0 is known.
  This program downsamples the data and then uses a minimization routine
  to find the best wavelength-dependent b and c coefficients by
  minimizing the difference between the calculated and observed data for
  multiple viewing geometries and multiple grain sizes simultaneously.
  """
  # TODO: downsample? would need to adjust spectra, k, n
  wave = spectra[0][:,0]
  actuals = [traj[:,1] for traj in spectra]

  # TODO: solve for k, b, c, s, D (see MasterPhaseWrapper_Byt2)
  # could use alternating minimizer here, like optimize_global_k
  solutions = []
  return solutions
