from __future__ import division, print_function
import copy
import numpy as np

__all__ = ['get_hapke_model']

#The base Hapke Model
def get_hapke_model(phase_fn='legendre', scatter='isotropic'):
  if phase_fn == 'legendre':
    phase_mixin = LegendrePhaseMixin
  elif phase_fn == 'dhg':
    phase_mixin = DoubleHeyneyGreensteinPhaseMixin
  elif phase_fn == 'constant':
    phase_mixin = ConstantPhaseMixin
  else:
    raise ValueError('Invalid phase_fn: %r' % phase_fn)
  if scatter == 'isotropic':
    scatter_mixin = IsotropicMixin
  elif scatter == 'lambertian':
    scatter_mixin = LambertianMixin
  else:
    raise ValueError('Invalid scatter: %r' % scatter)

  class HapkeModel(_BaseHapke, phase_mixin, scatter_mixin):
    pass

  return HapkeModel


class _BaseHapke(object):
  def __init__(self, refraction_index, opposition_surge):
    self._init_refraction(refraction_index)
    # later calculations need B(g) + 1
    self.Bg1 = opposition_surge + 1

  def _init_refraction(self, n):
    self.n = n = np.asarray(n)
    numer = (n - 1) ** 2
    denom = (n + 1) ** 2
    # approximate surface reflection coefficient S_E
    self.Se = numer / denom + 0.05
    # approximate internal scattering coefficient S_I
    self.Si = 1.014 - 4 / (n * denom)

  def radiance_coeff(self, wave, dispersion, internal_scatter, grain_size,
                     thetai, thetae, phase_params, scatter_params):
    scat_eff = self.scattering_efficiency(dispersion, wave, grain_size,
                                          internal_scatter)
    Pg = self.single_particle_phase(thetai, thetae, *phase_params)
    return self.radiance(thetai, thetae, scat_eff, Pg, *scatter_params)

  def copy(self, refraction_index=None, opposition_surge=None):
    model = copy.copy(self)
    if refraction_index is not None:
      model._init_refraction(refraction_index)
    if opposition_surge is not None:
      model.Bg1 = opposition_surge + 1
    return model

  def scattering_efficiency(self, wave, k, D, s):
    Alpha = (4 * np.pi * k) / wave
    Alpha_s = Alpha + s
    tmp = np.sqrt(Alpha/(Alpha_s))
    ri = (1-tmp) / (1+tmp)
    tmp = np.exp(-D * np.sqrt(Alpha*Alpha_s))
    THETA = (ri + tmp) / (1 + ri*tmp)
    return self.Se + (1-self.Se)*(((1-self.Si)*THETA)/(1 - self.Si*THETA))

  def _r0(self, scat_eff):
    gamma = np.sqrt(1 - scat_eff)
    return (1-gamma) / (1+gamma)

  def _Hu(self, scat_eff, u, r0=None):
    if r0 is None:
      r0 = self._r0(scat_eff)
    return 1/(1 - u*scat_eff*(r0+np.log((1+u)/u)*(0.5 - r0*u)))

  def _Hu_Hu0(self, scat_eff, u, u0):
    r0 = self._r0(scat_eff)
    Hu = self._Hu(scat_eff, u=u, r0=r0)
    Hu0 = self._Hu(scat_eff, u=u0, r0=r0)
    return Hu, Hu0


class LegendrePhaseMixin(object):
  def single_particle_phase(self, thetai, thetae, b, c):
    # two-term legendre polynomial phase function P(g)
    cosg = np.cos(np.abs(thetae - thetai))
    return 1 + b * cosg + c * (1.5*(cosg**2)-0.5)


class DoubleHeyneyGreensteinPhaseMixin(object):
  def single_particle_phase(self, thetai, thetae, b, c):
    # double Heyney-Greenstein phase function P(g)
    x0 = b * b + 1
    x1 = 2 * b * np.cos(np.abs(thetae - thetai))
    x3 = (x0 - x1) ** 1.5
    x4 = (x0 + x1) ** 1.5
    return (x0 - 2) * (x3*(c-1) - x4*(c+1)) / (2 * x3 * x4)


class ConstantPhaseMixin(object):
  def single_particle_phase(self, thetai, thetae):
    return 1


class LambertianMixin(object):
  needs_isow = False

  def radiance(self, thetai, thetae, scat_eff, Pg):
    u, u0 = np.cos((thetai, thetae))
    Hu, Hu0 = self._Hu_Hu0(scat_eff, u, u0)
    tmp = Pg * self.Bg1 + Hu * Hu0 - 1
    return tmp * (scat_eff/4.) / (u + u0)


class IsotropicMixin(object):
  needs_isow = True

  def radiance(self, thetai, thetae, scat_eff, Pg, ff, isow):
    u, u0 = np.cos((thetai, thetae))
    isoHu, isoHu0 = self._Hu_Hu0(isow, u, u0)

    # calculate the porosity constant for equant particles (K in Hapke 2008)
    tmp = -1.209 * ff**(2./3)
    porosity_constant = np.log1p(tmp) / tmp

    # perform the change of variables to account for porosity
    # see Hapke 2012b (the book) for details.
    Hu, Hu0 = self._Hu_Hu0(scat_eff, u / porosity_constant,
                           u0 / porosity_constant)

    tmp = Pg * self.Bg1 + Hu * Hu0 - 1
    numer = porosity_constant * tmp * scat_eff
    denom = isow * isoHu * isoHu0
    return numer / denom
