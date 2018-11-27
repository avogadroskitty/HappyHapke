#!/usr/bin/env python
"""
This program runs a Hapke radiative transfer model to derive optical
constants n and k from reflectance data. This version of the program
assumes that you have used a calibrated spectralon standard.
"""
from __future__ import division, print_function
import numpy as np
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt

import analysis
from hapke_model import get_hapke_model


def parse_args():
  data_dir = os.path.relpath(os.path.join(os.path.dirname(__file__),
                                          '..', 'data'))
  ap = ArgumentParser(description=__doc__,
                      formatter_class=ArgumentDefaultsHelpFormatter)

  g = ap.add_argument_group('Data Files', description="""
    Input MAT-files for your three grain sizes (VNIR spectral names).
    Data should be 2-column (wavelength, data) with no header.
    Can be in nm or microns but not wavenumbers.""")
  g.add_argument('--small-file', default=os.path.join(data_dir, 'kjs.mat'),
                 help='Small grain size')
  g.add_argument('--medium-file', default=os.path.join(data_dir, 'kjm.mat'),
                 help='Medium grain size')
  g.add_argument('--large-file', default=os.path.join(data_dir, 'kjb.mat'),
                 help='Large grain size')
  g.add_argument('--mir-dispersion-k', help='MIR k data',
                 default=os.path.join(data_dir, 'kjar_110813_disp_k.mat'))
  g.add_argument('--mir-dispersion-v', help='MIR wavelength data',
                 default=os.path.join(data_dir, 'kjar_110813_disp_v.mat'))

  g = ap.add_argument_group('Calibration Files', description="""
    Only needed for --scatter-type=isotropic""")
  g.add_argument('--specwave-file', help='Matlab MAT-file with specwave data',
                 default=os.path.join(data_dir, 'specwave2.mat'))
  g.add_argument('--calspec-file', help='Matlab MAT-file with calspec data',
                 default=os.path.join(data_dir, 'calspecw2.mat'))

  g = ap.add_argument_group('Usable Range', description="""
    Usable range of wavelengths (in microns) - if you have bad data in the file,
    it will make the slope and intercept calculations wonky.
    LOW MUST BE AT LEAST ONE POINT IN FROM END OF DATA""")
  g.add_argument('--low', default=0.32, help='Lower bound.')
  g.add_argument('--high', default=2.55, help='Upper bound.')

  g = ap.add_argument_group('Known Variables')
  g.add_argument('--incident-angle', dest='thetai', default=-30,
                 help='Incident angle, degrees.')
  g.add_argument('--emission-angle', dest='thetae', default=0,
                 help='Emission angle, degrees.')
  g.add_argument('--average-n', dest='n1', default=(1.714+1.8175)/2,
                 help='Average n, usually at sodium D line.')
  g.add_argument('--anchor', default=0.58929,
                 help='Anchor wavelength at which --average-n was determined.')
  g.add_argument('--opposition-surge', dest='Bg', default=0,
                 help='Opposition surge, can be 0 if angular diff > 15.')
  g.add_argument('--extinction-efficiency', dest='QE', default=1,
                 help=('Extinction efficiency, '
                       'set to 1 for closely-spaced particles.'))
  g.add_argument('--uv', dest='UV', default=0.301,
                 help=('UV wavelength point for extrapolation. '
                       'We only need about 10 points but more is better.'))

  g = ap.add_argument_group('Hapke Model Parameters', description="""
    Choices about the parameterization of the model.""")
  g.add_argument('--phase-function', choices=('legendre', 'dhg', 'constant'),
                 default='legendre', help='Type of phase function to use.')
  g.add_argument('--scatter-type', choices=('isotropic', 'lambertian'),
                 default='isotropic', help='Type of scattering to use.')

  g = ap.add_argument_group('Radiance Parameters', description="""
    Values for each of small, medium, and large grain sizes.""")

  g.add_argument('--grain-size', dest='D', nargs=3, type=float,
                 default=[45, 63, 90], help='Grain size guesses.')
  g.add_argument('--grain-size-lower', dest='lowD', nargs=3, type=float,
                 default=[21, 30, 50], help='Grain size lower bounds.')
  g.add_argument('--grain-size-upper', dest='upD', nargs=3, type=float,
                 default=[106, 150, 180], help='Grain size lower bounds.')

  g.add_argument('--internal-scattering', dest='s', nargs=3, type=float,
                 default=[0, 0, 0], help='Internal scattering param guesses.')
  g.add_argument('--internal-scattering-lower', dest='lows', nargs=3,type=float,
                 default=[0, 0, 0], help='Internal scattering lower bounds.')
  g.add_argument('--internal-scattering-upper', dest='ups', nargs=3, type=float,
                 default=[0.06, 0.06, 0.06],
                 help='Internal scattering upper bounds.')

  g.add_argument('--legendre-b', dest='b', nargs=3, type=float,
                 default=[0.1, 0.1, 0.1],
                 help='Guesses for Legendre polynomial coefficient b.')
  g.add_argument('--legendre-b-lower', dest='lowb', nargs=3, type=float,
                 default=[-1.7, -1.7, -1.7],
                 help='Lower bounds for Legendre polynomial coefficient b.')
  g.add_argument('--legendre-b-upper', dest='upb', nargs=3, type=float,
                 default=[1.7, 1.7, 1.7],
                 help='Upper bounds for Legendre polynomial coefficient b.')

  g.add_argument('--legendre-c', dest='c', nargs=3, type=float,
                 default=[0.3, 0.3, 0.3],
                 help='Guesses for Legendre polynomial coefficient c.')
  g.add_argument('--legendre-c-lower', dest='lowc', nargs=3, type=float,
                 default=[-1, -1, -1],
                 help='Lower bounds for Legendre polynomial coefficient c.')
  g.add_argument('--legendre-c-upper', dest='upc', nargs=3, type=float,
                 default=[1, 1, 1],
                 help='Upper bounds for Legendre polynomial coefficient c.')

  g.add_argument('--filling-factor', dest='ff', nargs=3, type=float,
                 default=[0.00000000001, 0.00000000001, 0.00000000001],
                 help="""Filling factor guesses. If you cannot define it,
                         set it to 1e-17 but according to Hapke 2008,
                         absence of a good estimate this term can result in
                         k being off by as much as a factor of 2""")

  g.add_argument('--k-lower', dest='lowk', type=float, default=0,
                 help='Lower bound for k.')
  g.add_argument('--k-upper', dest='upk', type=float, default=.1,
                 help='Upper bound for k.')

  ap.add_argument('--debug-plots', action='store_true',
                  help='Show plots for debugging purposes.')
  return ap.parse_args()


def main():
  args = parse_args()
  print('Preparing variables')
  HapkeModel = get_hapke_model(phase_fn=args.phase_function,
                               scatter=args.scatter_type)
  hapke = HapkeModel(np.deg2rad(args.thetai), np.deg2rad(args.thetae),
                     args.n1, args.Bg)
  files = dict(sml=args.small_file, med=args.medium_file, big=args.large_file)
  params = {}
  for i, key in enumerate(('file1', 'file2', 'file3')):
    params[key] = (args.b[i], args.c[i], args.ff[i], args.s[i], args.D[i])

  if hapke.needs_isow:
    # initialize isow as the mean of a fixed range
    specwave = analysis.loadmat_single(args.specwave_file).ravel()
    calspec = analysis.loadmat_single(args.calspec_file).ravel()
    isoind1, isoind2 = np.searchsorted(specwave, (0.5, 1.25))
    hapke.set_isow(calspec[isoind1:isoind2].mean())

  # section 1
  print('Running section 1')
  spectra = {}
  for key, infile in files.items():
    traj = analysis.loadmat_single(infile)
    spectra[key] = analysis.preprocess_traj(traj, args.low, args.high, args.UV)

  # sections 2, 3, 4
  ks = {}
  for key, traj in spectra.items():
    print('Running sections 2,3,4 (MasterHapke1_PP: %s)' % key)
    ks[key] = analysis.MasterHapke1_PP(hapke, traj, *params[key],
                                       debug_plots=args.debug_plots)

  # section 5 isn't worth porting
  print('Skipping section 5')

  # section 6: iterative minimizations
  print('Running section 6 (MasterHapke2_PP)')
  if hapke.needs_isow:
    specwave, calspec = analysis.prepare_spectrum(specwave, calspec, args.UV,
                                                  args.high)
    hapke.set_isow(calspec)
  # use the medium-grain k as an initial guess
  k = ks['file2']
  # XXX: this takes too long, skip it
  # guesses = np.concatenate((args.b, args.c, args.s, args.D, k))
  # lb = np.concatenate((args.lowb, args.lowc, args.lows, args.lowD,
  #                      np.zeros_like(k) + args.lowk))
  # ub = np.concatenate((args.upb, args.upc, args.ups, args.upD,
  #                      np.zeros_like(k) + args.upk))
  # solutions = analysis.MasterHapke2_PP(hapke, spectra, guesses, lb, ub,
  #                                      args.ff, tr_solver='lsmr', verbose=2)

  # section 7/8: graphs the parameters from the previous section
  if args.debug_plots:
    # print 'Running sections 7,8 (plotting %d solutions)' % len(solutions)
    # see HapkeEval1_PP.m
    # plot initial guesses -> solved values for b, c, s, D, and k
    # TODO
    # see HapkeEval3_PP.m
    # plot given reflectances (spectra) vs solved rcs
    pass

  # section 9: add in your MIR data
  # If you do not have MIR data (not recommended), you can skip this step.
  # If you do have MIR data, use the DISPERSION programs on the website to get
  # k data for your sample through the MIR.
  wave = spectra['file2'][:,0]
  if all(os.path.exists(f) for f in (args.mir_dispersion_k,
                                     args.mir_dispersion_v)):
    print('Running section 9 (MasterKcombine)')
    combined = analysis.MasterKcombine(args.mir_dispersion_k,
                                       args.mir_dispersion_v, wave, k)
    if args.debug_plots:
      plt.figure()
      plt.plot(10000/combined[:,0], combined[:,1])
      plt.title('MasterKcombine')
  else:
    print('Skipping section 9 (MasterKcombine): MIR data not found')
    combined = np.column_stack((10000/wave, k))

  # section 10: singly subtractive Kramers Kronig calculation
  print('Running section 10 (MasterSSKK)')
  res = analysis.MasterSSKK(combined, args.n1, args.anchor)
  if args.debug_plots:
    fig, ax = plt.subplots()
    ax.plot(10000/res[:,0], res[:,1])
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('n')

  if args.debug_plots:
    plt.show()


if __name__ == '__main__':
  main()
