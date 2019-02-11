from .spheres_mueller import MuellerMatrixMie, MuellerMatrixMieScalar, CalcScatteringProps, MAX_LEG, MAX_DEG, np, PI4, CalcScatteringProps1
import pprint

pp = pprint.PrettyPrinter(compact=True)
#print("---== Initialize Rayleigh module ==---")
RayEv = np.zeros((MAX_LEG+1,6), dtype='float64')
RayEv[:3,0] = (1.0, 0.0, 0.5)
RayEv[:3,1] = (-0.5, 0.0, 0.5)
#RayEv[:3,2] = (0.0, 1.5, 0.0)
#RayEv[:3,3] = (0.0, 0.0, 0.0)
RayEv[:3,4] = (1.0, 0.0, 0.5)
#RayEv[:3,5] = (0.0, 1.5, 0.0)

#pp.pprint(RayEv)

def AotRayleigh(wl):
  """
  Расчет молекулярной оптической толщи
  """
  return 0.008569 / (wl**4.0) * \
    (1.0 + 0.0113/(wl**2.0) + 0.00013/(wl**4.0))



def PrepareScatFile(EvA, taua, ssa, wavelen):
  r, c = EvA.shape
  ret = np.zeros_like(EvA)
  tmp = np.zeros_like(EvA)
  
  taua_s = ssa*taua
  taum = AotRayleigh(wavelen)
  ret = (EvA*taua_s+taum*RayEv)/(taua_s+taum)
  taut = taum+taua
  ssat = (taua_s + taum) / (taua + taum)
  
  with open('scat_file', 'wt') as fout:
    fout.write(f"{taut:12.4f}\n")
    fout.write(f"{(taut*ssat):12.4f}\n")
    fout.write(f"{ssat:12.4f}\n")
    fout.write(f"{r-1:12d}\n")
    
    for i in range(r):
      fout.write(f"{i:7d}{ret[i,0]:14.6e}{ret[i,1]:14.6e}"+\
        f"{ret[i,2]:14.6e}{ret[i,3]:14.6e}"+\
        f"{ret[i,4]:14.6e}{ret[i,5]:14.6e}\n")
    fout.close()
  return taut,ssat,ret