from __future__ import division, print_function
import copy
import numpy as np

class HapkeModel(object):
  def __init__(self, incident_angle, emission_angle, refraction_index,
               opposition_surge, phase_mixing, scatter_mixing):
    self._init_angles(incident_angle, emission_angle)
    self._init_refraction(refraction_index)
    # later calculations need B(g) + 1
    self.Bg1 = opposition_surge + 1
    self.n1 = refraction_index
    self.phase_mixing = phase_mixing
    self.scatter_mixing = scatter_mixing
    self.needs_isow = False if scatter_mixing == 'lambertian' else True 

  def _init_angles(self, thetai, thetae):
    thetai = np.asarray(thetai)
    thetae = np.asarray(thetae)
    self.cosg = np.cos(np.abs(thetae - thetai))
    self.u0 = np.cos(thetai)
    self.u = np.cos(thetae)

    # np.asarray can handle both scalars and arrays - it will not rearray an array in a wierd way
  def _init_refraction(self, n):
    self.n = n = np.asarray(n)
    numer = (n - 1) ** 2
    denom = (n + 1) ** 2
    # approximate surface reflection coefficient S_E
    self.Se = numer / denom + 0.05
    # approximate internal scattering coefficient S_I
    self.Si = 1.014 - 4 / (n * denom)

  def copy(self, incident_angle=None, emission_angle=None,
           refraction_index=None, opposition_surge=None):
    model = copy.copy(self)
    if incident_angle is not None:
      model._init_angles(incident_angle, emission_angle)
    if refraction_index is not None:
      model._init_refraction(refraction_index)
    if opposition_surge is not None:
      model.Bg1 = opposition_surge + 1
    return model

  def scattering_efficiency(self, k, wave, D, s, n):
    
    self._init_refraction(n)

    Alpha = (4 * np.pi * k) / wave
    Alpha_s = Alpha + s
    tmp = np.sqrt(Alpha/(Alpha_s))
    ri = (1-tmp) / (1+tmp)
    tmp = np.exp(-D * np.sqrt(Alpha*Alpha_s))
    #Hapke1993 equation 11.15a
    THETA = (ri + tmp) / (1 + ri*tmp)
    #Hapke1993 equation 11.14 for w aka single scattering albedo
    return self.Se + (1-self.Se)*(((1-self.Si)*THETA)/(1 - self.Si*THETA))

  def _r0(self, scat_eff):
    gamma = np.sqrt(1 - scat_eff)
    return (1-gamma) / (1+gamma)

  def _Hu(self, scat_eff, u, r0=None):
    if r0 is None:
      r0 = self._r0(scat_eff)
      #Hapke 1993 equation 8.57
      #H(x) = {1/{1-[1-gamma]*x*[r0+(1-0.5*r0-r0*x)*ln((1+x)/x)]}}
      #Cj had return 1/(1 - u*scat_eff*(r0+np.log((1+u)/u)*(0.5 - r0*u)))
      #this is not the same!
    tmp_gamma = np.sqrt(1 - scat_eff)
    return 1/(1-(1-tmp_gamma)*u*(r0 + (1 - 0.5*r0 - u*r0)*np.log((1 + u)/u)))

  def _Hu_Hu0(self, scat_eff, u, u0):
    r0 = self._r0(scat_eff)
    Hu = self._Hu(scat_eff, u=u, r0=r0)
    Hu0 = self._Hu(scat_eff, u=u0, r0=r0)
    return Hu, Hu0

  def single_particle_phase(self, b, c):
      if self.phase_mixing == 'legendre':
        # two-term legendre polynomial phase function P(g)
        return 1 + b * self.cosg + c * (1.5*(self.cosg**2)-0.5)
      elif self.phase_mixing == 'dhg':
        # double Heyney-Greenstein phase function P(g)
        x0 = b * b + 1
        x1 = 2 * b * self.cosg
        x3 = (x0 - x1) ** 1.5
        x4 = (x0 + x1) ** 1.5
        return (x0 - 2) * (x3*(c-1) - x4*(c+1)) / (2 * x3 * x4)
      elif self.phase_mixing == 'constant':
        return 1
      else:
        raise ValueError('Invalid phase_fn: %r' % phase_fn)
  
  def radiance_coeff(self, scat_eff, b, c, ff=None):
      if self.scatter_mixing == 'isotropic':                
          # calculate the porosity constant for equant particles (K in Hapke 2008)
          tmp = -1.209 * ff**(2./3)
          PoreK = np.log1p(tmp) / tmp

          # perform the change of variables to account for porosity outside of H(u)
          # and H(u0) equations for simplicity, i.e. H(u) will become H(u/PoreK)
          # see Hapke 2012b (the book) for details.
          uK = self.u / PoreK
          u0K = self.u0 / PoreK
          Hu, Hu0 = self._Hu_Hu0(scat_eff, uK, u0K)

          Pg = self.single_particle_phase(b, c)
          tmp = Pg * self.Bg1 + Hu*Hu0 - 1
          numer = PoreK * tmp * scat_eff
          return numer / self.rc_denom

      elif self.scatter_mixing == 'lambertian':
          #Lambertian Mixing
          Pg = self.single_particle_phase(b, c)
          Hu, Hu0 = self._Hu_Hu0(scat_eff, self.u, self.u0)
          tmp = Pg * self.Bg1 + Hu*Hu0 - 1
          #Hapke1993 equation 10.4
          return tmp * (scat_eff/4.) / (self.u + self.u0)
      else:
        raise ValueError('Invalid scatter: %r' % scatter)

  def set_isow(self, isow):
    self.isow = isow
    self.isoHu, self.isoHu0 = self._Hu_Hu0(isow, self.u, self.u0)
    self.rc_denom = self.isow * self.isoHu0 * self.isoHu
