#define methods to calculate scattering props for nonspherical particles

import numpy as np


MAX_LEG = 30
MAX_DEG = 64
PI4 = 4*np.pi


def CalcScatteringPropsAggl(F, wavelen, X, mu, Csca, Cext, Ev, nleg=0):
  """
      Вычисляет матрицу рассеяния и оптические коэффициенты для степенного распределения атмосферного аэрозоля по размерам. В случае необходимости, возвращает разложение по плиномам Лежандра.
  """
#  f = lambda r, gamma:  r**gamma
#  norm = (f(r1, gamma+1)-f(r0, gamma+1))/(gamma+1)
  
  k = 2*np.pi/wavelen
  
  r = X / k
  f = F(r)
  s = np.pi*r*r
  muel_tmp = np.array([Ev[i,:,:,:]*f[i] for i in range(Csca.shape[0])])
  #print(muel_tmp.shape)
  EvA = np.trapz(muel_tmp, r, axis=0)
  sca_tot = np.trapz(Csca*f, r)
  ext_tot = np.trapz(Cext*f, r)
  EvA = EvA / sca_tot
  zero = np.zeros(EvA.shape[0])
  
  evans_tot = np.c_[EvA[:,0,0], EvA[:,0,1], zero, zero, EvA[:,0,0], zero]
  
  #print(np.trapz(evans_tot[:,0], mu)*PI4)
  
  if nleg>3:
    lv=np.polynomial.legendre.legvander(mu, nleg)
    evans_tot, residuals, rank, s=np.linalg.lstsq(lv, PI4*evans_tot, rcond=None)
  #print(evans_tot[0,0], sca_tot/ext_tot)
  return evans_tot, sca_tot/ext_tot