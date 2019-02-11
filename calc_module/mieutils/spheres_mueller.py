import miepython
import numpy as np


MAX_LEG = 30
MAX_DEG = 64
PI4 = 4*np.pi
def MuellerMatrixMieScalar(x, m, mu):
  """
  Description:
  ~~~~~~~~~~~~
  x - size parameter 2*pi*r/lambda
  m - refractive index
  mu- cosine of scattering angle
  """
  
  qext, qsca, qback, g = miepython.mie(m, x)
  S1, S2 = miepython.mie_S1_S2(m, x, mu)
  nr, nc = len(mu), 6
  muel = np.zeros((nr, nc), dtype='float64')
  
  muel[:,0] = 0.5*(np.abs(S2)**2+np.abs(S1)**2)
  muel[:,1] = 0.5*(np.abs(S2)**2-np.abs(S1)**2)
  #muel[:,2] = (S2.conjugate()*S1).real
  #muel[:,3] = (S2*S1.conjugate()).imag
  muel[:,4] = muel[:,0]
  #muel[:,5] = muel[:,2]
  
  return muel, qext, qsca, qback, g
  
def MuellerMatrixMie(x, m, mu):
  """
  Description:
  ~~~~~~~~~~~~
  x - size parameter 2*pi*r/lambda, may be any iterable object
  m - refractive index
  mu- cosine of scattering angle
  """
  
  if hasattr(x, '__iter__'):
    nx = len(x)
    nr = len(mu)
    muel = np.zeros((nx, nr, 6), dtype='float64')
    qext = np.zeros(nx)
    qsca = np.zeros_like(qext)
    qback = np.zeros_like(qext)
    g = np.zeros_like(qext)
    
    for i in range(nx):
      muel[i,:,:], qext[i], qsca[i], qback[i], g[i] =\
                  MuellerMatrixMieScalar(x[i], m, mu)
  else:
    muel, qext, qsca, qback, g = MuellerMatrixMieScalar(x, m, mu)
  
  return muel, qext, qsca, qback, g
  

def CalcScatteringProps(r0, r1, gamma, wavelen, midx, mu, nleg=0):
  """
      Вычисляет матрицу рассеяния и оптические коэффициенты для степенного распределения атмосферного аэрозоля по размерам. В случае необходимости, возвращает разложение по плиномам Лежандра.
  """
  f = lambda r, gamma:  r**gamma
  norm = (f(r1, gamma+1)-f(r0, gamma+1))/(gamma+1)
  k = 2*np.pi/wavelen
  ri, wi = np.polynomial.legendre.leggauss(MAX_DEG)
  ri = (r1-r0)/2.0*ri+(r1+r0)/2.0
  wi = (r1-r0)/2.0*wi
  #print(wi[0], wi[-1])
  
  ext=sca=back=0.0
  evans_tot = np.zeros((len(mu), 6), dtype='float64')
  for i in range(MAX_DEG):
    r_i = ri[i]
    f_i = f(r_i, gamma) / norm
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
  
  
def CalcScatteringProps1(F, wavelen, midx, mu, nleg=0):
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