import numpy as np


__all__=['PowerLaw', 'LogNormal', 'LogNormal2']

class BaseDistr:
  
  def __init__(self, r0, r1):
    self.r0 = r0
    self.r1 = r1
    
  def __call__(self, r):
    return ((r>=self.r0)&(r<=self.r1))*1.0


class PowerLaw(BaseDistr):
  
  def __init__(self, r0, r1, gamma):
    super(PowerLaw, self).__init__(r0, r1)
    self.gamma = gamma
    self.norm = (r1**(gamma+1.0) - r0**(gamma+1.0))/(gamma+1.0)
  
  
  def __call__(self, r):
    tmp = super(PowerLaw, self).__call__(r)
    return tmp*(r**self.gamma)/self.norm
    
class LogNormal(BaseDistr):
  
  def __init__(self, r0, r1, rm, s):
    super(LogNormal, self).__init__(r0, r1)
    self.rm = rm
    self.s = s
    
  def __call__(self, r):
    sqrt2pi = np.sqrt(np.pi*2.0)
    tmp = super(LogNormal, self).__call__(r)
    mu = np.log(self.rm)
    sigma = np.log(self.s)
    
    ret = (np.exp(-0.5*((np.log(r) - mu) / sigma)**2) \
      / (r * sigma * np.sqrt(2 * np.pi)))*tmp
    return ret
    
class LogNormal2:
  """ Bimodal lognormal distribution as superposition of two single
  mode distributions with a parameter F.
  """
  
  def __init__(self, r0, r1, rm1, s1, rm2, s2, f):
    self.r0, self.r1 = r0, r1
    self.L1 = LogNormal(r0, r1, rm1, s1)
    self.L2 = LogNormal(r0, r1, rm2, s2)
    self.F = f
  
  def __call__(self, r):
    L1 = self.L1(r)
    L2 = self.L2(r)
    return self.F*L1+(1.0-self.F)*L2

  