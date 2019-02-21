#define methods to calculate scattering props for nonspherical particles

import numpy as np


MAX_LEG = 30
MAX_DEG = 64
PI4 = 4*np.pi


def CalcScatteringPropsAggl(F, wavelen, midx, mu, nleg=0):
  """
      Вычисляет матрицу рассеяния и оптические коэффициенты для степенного распределения атмосферного аэрозоля по размерам. В случае необходимости, возвращает разложение по плиномам Лежандра.
  """
#  f = lambda r, gamma:  r**gamma
#  norm = (f(r1, gamma+1)-f(r0, gamma+1))/(gamma+1)
  
  k = 2*np.pi/wavelen
  ri, wi = np.polynomial.legendre.leggauss(MAX_DEG)
  ri = (F.r1-F.r0)/2.0*ri+(F.r1+F.r0)/2.0
  wi = (F.r1-F.r0)/2.0*wi
  #print(wi[0], wi[-1])
  
  ext=sca=back=0.0
  evans_tot = np.zeros((len(mu), 6), dtype='float64')
  for i in range(MAX_DEG):
    r_i = ri[i]
    f_i = F(r_i)
    s_i = np.pi*r_i*r_i
    x_i = r_i*k
    
    evans, qext, qsca, qback, _ = MuellerMatrixMieScalar(x_i, midx, mu)
    
    csca_i = qsca*s_i*f_i*wi[i]
    ext = ext + qext*s_i*f_i*wi[i]
    sca = sca + csca_i
    back = back + qback*s_i*f_i*wi[i]
    evans_tot += evans*csca_i
    

  evans_tot/=sca
 
  
  if nleg>3:
    lv=np.polynomial.legendre.legvander(mu, nleg)
    evans_tot, residuals, rank, s=np.linalg.lstsq(lv, PI4*evans_tot, rcond=None)
  return evans_tot, ext*1e-12, sca*1e-12, back*1e-12, sca/ext