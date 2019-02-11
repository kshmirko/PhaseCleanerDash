import subprocess as sp
from dataclasses import dataclass
from typing import List
import numpy as np
import os
import sys
from collections import namedtuple

RtResult = namedtuple('RtResult',['mu', 'I', 'Q'])

@dataclass
class RtCode:
  solar_zenith: float
  direct_flux: float
  ground_albedo: float
  num_stokes: int = 2
  num_quad: int = 32
  type_quad: str = 'G'
  order_azim_exp: int = 8
  layfile: str = 'atmos.lay'
  delta_m: str = 'N'
  src_code: int = 1
  ground_temp: float = 0.0
  ground_type: str = 'L'
  sky_temp: float = 0.0
  wavelength: float = 0.750
  output_rad_units: str = 'W'
  output_pol: str = 'IQ'
  nout_levels: int = 1
  out_level: int = 2
  nout_azim: int = 2
  out_fname: str = 'rt3.out' 
  
  
  def run(self):
    if not os.path.exists('./rt3-code'):
        raise Exception("Can't file executable model filename in current directory")
    self.subproc = sp.Popen('./rt3-code', stdin=sp.PIPE, stdout=sp.PIPE)
    self.subproc.stdin.write(f"{self.num_stokes}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.num_quad}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.type_quad}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.order_azim_exp}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.layfile}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.delta_m}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.src_code}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.direct_flux}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.solar_zenith}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.ground_temp}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.ground_type}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.ground_albedo}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.sky_temp}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.wavelength}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.output_rad_units}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.output_pol}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.nout_levels}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.out_level}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.nout_azim}\n".encode())
    self.subproc.stdin.flush()
    self.subproc.stdin.write(f"{self.out_fname}\n".encode())
    self.subproc.stdin.flush()
    #print(self.subproc.stdout.readlines())
    self.subproc.wait()
    #print("Rt3 Code completed!")
    
    # Immediately read the output from the program!!!
    # Just after the execution!!!!
    self.read_output()
  
  def read_output(self):
    """
    
    """
    self.z, phi, mu, I, Q =\
            np.loadtxt(self.out_fname, skiprows=11, unpack=True)
            
    idx = mu>0
    phi = phi[idx]
    mu = np.rad2deg(np.arccos(mu[idx])*np.cos(np.deg2rad(phi)))
    
    I = I[idx]
    Q = Q[idx]
    
    mu[self.num_quad:] = mu[2*self.num_quad:self.num_quad-1:-1]
    I[self.num_quad:] = I[2*self.num_quad:self.num_quad-1:-1]
    Q[self.num_quad:] = Q[2*self.num_quad:self.num_quad-1:-1]
    
    
    self.mu = self.solar_zenith-mu
    idx = self.mu>=0
    self.mu = np.cos(np.deg2rad(self.mu[idx]))
    self.I = I[idx]
    self.Q = Q[idx]
    self.mu = np.flip(self.mu[:])
    self.I = np.flip(self.I[:])
    self.Q = np.flip(self.Q[:])
    
    #self.result = RtResult(self.mu, self.I, self.Q)
    #self.mu = mu[-1::-1]
    #self.I = I[-1::-1]
    #self.Q = Q[-1::-1]
    

    
    