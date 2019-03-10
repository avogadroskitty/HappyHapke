from __future__ import division, print_function
import copy
import numpy as np
import math
from scipy import interpolate

class HapkeModel(object):
  def __init__(self, incident_angle, emission_angle, refraction_index,
               needs_bg, phase_mixing, scatter_mixing, hu_approx=False):
    # These angles are already in radians
    self._init_angles(incident_angle, emission_angle)
    self._init_refraction(refraction_index)
    # old: B(g) + 1 | New: Added backscatter function
    self.needs_bg = needs_bg
    self.n1 = refraction_index
    self.phase_mixing = phase_mixing
    self.scatter_mixing = scatter_mixing
    self.needs_isow = False if scatter_mixing == 'lambertian' else True 
    self.hu_approx = hu_approx

  def _init_angles(self, thetai, thetae):
    self.thetai = np.asarray(thetai)
    self.thetae = np.asarray(thetae)
    self.g = np.abs(thetae) + np.abs(thetai)
    self.cosg = np.cos(self.g)
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
    if incident_angle is not None and emission_angle is not None:
      model._init_angles(incident_angle, emission_angle)
    if refraction_index is not None:
      model._init_refraction(refraction_index)
    if opposition_surge is not None:
      model.needs_bg = opposition_surge
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
    if self.hu_approx: 
        if r0 is None:
            r0 = self._r0(scat_eff)
          #Hapke 1993 equation 8.57
          #H(x) = {1/{1-[1-gamma]*x*[r0+(1-0.5*r0-r0*x)*ln((1+x)/x)]}}
          #Cj had return 1/(1 - u*scat_eff*(r0+np.log((1+u)/u)*(0.5 - r0*u)))
          #this is not the same!
        tmp_gamma = np.sqrt(1 - scat_eff)
        val = 1/(1-(1-tmp_gamma)*u*(r0 + (1 - 0.5*r0 - u*r0)*np.log((1 + u)/u)))
        return val
    else: 
        w0_table = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 1.0])
        u_table = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
        #h = 21, 14 (u, w)
        h_table = np.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                      [1.00783, 1.01608, 1.02484, 1.03422, 1.04439, 1.05544, 1.06780, 1.0820, 1.0903, 1.0999, 1.1053, 1.1117, 1.1196, 1.1368],
                      [1.01238, 1.02562, 1.03989, 1.05535, 1.07241, 1.09137, 1.11306, 1.1388, 1.1541, 1.1722, 1.1828, 1.1952, 1.2111, 1.2474],
                      [1.01584, 1.03295, 1.05155, 1.07196, 1.09474, 1.12045, 1.15036, 1.1886, 1.2086, 1.2349, 1.2506, 1.2693, 1.2936, 1.3508],
                      [1.01864, 1.03893, 1.06115, 1.08577, 1.11349, 1.14517, 1.18253, 1.2286, 1.2570, 1.2914, 1.3123, 1.3373, 1.3703, 1.4503],
                      [1.02099, 1.04396, 1.06930, 1.09758, 1.12968, 1.16674, 1.21095, 1.2663, 1.3009, 1.3433, 1.3692, 1.4008, 1.4427, 1.5473],
                      [1.02300, 1.04829, 1.07637, 1.10789, 1.14391, 1.18587, 1.23643, 1.3006, 1.3411, 1.3914, 1.4224, 1.4604, 1.5117, 1.6425],
                      [1.02475, 1.05209, 1.08259, 1.11700, 1.15659, 1.20304, 1.25951, 1.3320, 1.3783, 1.4363, 1.4724, 1.5170, 1.5778, 1.7364],
                      [1.02630, 1.05546, 1.08811, 1.12516, 1.16800, 1.21861, 1.28063, 1.3611, 1.4129, 1.4785, 1.5197, 1.5709, 1.6414, 1.8293],
                      [1.02768, 1.05847, 1.09308, 1.13251, 1.17833, 1.23280, 1.30003, 1.3881, 1.4453, 1.5183, 1.5646, 1.6224, 1.7027, 1.9213],
                      [1.02892, 1.06117, 1.09756, 1.13918, 1.18776, 1.24581, 1.31796, 1.4132, 1.4758, 1.5560, 1.6073, 1.6718, 1.7621, 2.0128],
                      [1.03004, 1.06363, 1.10164, 1.14528, 1.19640, 1.25781, 1.33459, 1.4368, 1.5044, 1.5918, 1.6480, 1.7191, 1.8195, 2.1037],
                      [1.03106, 1.06587, 1.10538, 1.15087, 1.20436, 1.26893, 1.35009, 1.4590, 1.5315, 1.6259, 1.6869, 1.7647, 1.8753, 2.1941],
                      [1.03199, 1.06793, 1.10881, 1.15602, 1.21173, 1.27925, 1.36457, 1.4798, 1.5571, 1.6583, 1.7242, 1.8086, 1.9295, 2.2842],
                      [1.03284, 1.06982, 1.11198, 1.16080, 1.21858, 1.28888, 1.37815, 1.4995, 1.5814, 1.6893, 1.7600, 1.8509, 1.9822, 2.3740],
                      [1.03363, 1.07157, 1.11491, 1.16523, 1.22495, 1.29788, 1.39090, 1.5182, 1.6045, 1.7190, 1.7943, 1.8918, 2.0334, 2.4635],
                      [1.03436, 1.07319, 1.11763, 1.16935, 1.23091, 1.30631, 1.40291, 1.5358, 1.6265, 1.7474, 1.8274, 1.9313, 2.0833, 2.5527],
                      [1.03504, 1.07469, 1.12017, 1.17320, 1.23648, 1.31424, 1.41425, 1.5526, 1.6475, 1.7746, 1.8592, 1.9695, 2.1320, 2.6417],
                      [1.03567, 1.07610, 1.12254, 1.17681, 1.24171, 1.32171, 1.42497, 1.5685, 1.6675, 1.8008, 1.8898, 2.0065, 2.1795, 2.7306],
                      [1.03626, 1.07741, 1.12476, 1.18019, 1.24664, 1.32875, 1.43512, 1.5837, 1.6867, 1.8259, 1.9194, 2.0423, 2.2258, 2.8193],
                      [1.03682, 1.07864, 1.12685, 1.18337, 1.25128, 1.33541, 1.44476, 1.5982, 1.7050, 1.8501, 1.9479, 2.0771, 2.2710, 2.9078]])
        fn = interpolate.interp2d(w0_table, u_table, h_table.T.flatten(), kind='linear')
        val = fn(scat_eff, u)[:,0]
        return val

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
  
  def radiance_coeff(self, scat_eff, b, c, ff=1e-11, b0=None, h=None):
      self.Bg1 = self.backscatter(b0,h) + 1 if self.needs_bg else 1
      # calculate the porosity constant for equant particles (K in Hapke 2008)
      PoreK = (-np.log(1 - 1.209 * ff**(2/3)))/(1.209 * ff**(2/3))
      # perform the change of variables to account for porosity outside of H(u)
      # and H(u0) equations for simplicity, i.e. H(u) will become H(u/PoreK)
      # see Hapke 2012b (the book) for details.
      uK = self.u / PoreK
      u0K = self.u0 / PoreK
      Hu, Hu0 = self._Hu_Hu0(scat_eff, uK, u0K)
      Pg = self.single_particle_phase(b, c)         
      tmp = Pg * self.Bg1 + Hu*Hu0 - 1   

      if self.scatter_mixing == 'isotropic':                
          numer = PoreK * ( scat_eff / 4. * math.pi) * (self.u / (self.u + self.u0)) * tmp
          return numer / self.rc_denom

      elif self.scatter_mixing == 'lambertian':
          #Lambertian Mixing
          #Hapke1993 equation 10.4
          numer = PoreK * ( scat_eff / 4.) * ( 1 / (self.u + self.u0)) * tmp
          return numer
      else:
        raise ValueError('Invalid scatter: %r' % scatter)

  def set_isow(self, isow):
    self.isow = isow
    self.isoHu, self.isoHu0 = self._Hu_Hu0(isow, self.u, self.u0)
    # ((isow./(4*pi))*(u/(u+u0)).*((1)+(isoHu0.*isoHu)-1));
    self.rc_denom = (self.isow / ( 4. * math.pi)) * (self.u / (self.u + self.u0)) * (1 + (self.isoHu0 * self.isoHu) - 1) 
         
  def backscatter(self, b0, h): 
      # Bg = B0/[1+(tan(g/2))/h] 
      if b0 is not None and h is not None:
          return b0 / ( 1 + ((np.tan(self.g / 2))/h)) 
      else:
          return 1
