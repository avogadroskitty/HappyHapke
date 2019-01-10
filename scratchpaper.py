

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

