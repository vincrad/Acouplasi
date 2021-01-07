#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:49:08 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
from Fluid import Fluid

# =============================================================================
# class Cavity(tr.HasTraits):
#     '''
#     Class to define a cavity for a plate resonator lining.
#     '''
#     # height of cavity
#     hc = tr.Float()
#     
#     fluid = tr.Instance(Fluid)
#     
#     def modes(self, length, depth, Lc, Mc, Nc):
#         # Parameter müssen aus PlateResonatorLining bekannt sein
#         # function which returns the modal sound field of the cavity
#         pass
# =============================================================================


class Cavity(tr.HasTraits):
    
    # geometry of the cavity
    length = tr.Float()
    depth = tr.Float(default_value=0)
    height = tr.Float()
    
    # loss factor
    zetars = tr.Float(default_value=0.1)
    
    # cavity modes
    R = tr.Instance(np.ndarray)
    S = tr.Instance(np.ndarray)
    
    # medium
    
    medium = tr.Instance(Fluid)
    
    
    def delta(self, r, ref=0):
        
        if r == ref:
            
            delta = 1
            
        else:
            
            delta = 0
            
        return delta
    
    
    def kappa(self, r, s):
        
        kappa = np.sqrt((r*np.pi/self.length)**2+(s*np.pi/self.height)**2)
        
        return kappa
    
    
    def ModImp(self, J, L, freq):
        
        Zc = np.zeros((len(J), len(L), len(freq)), dtype=complex)
        
        omega = 2*np.pi*freq
        
        k0 = omega*self.medium.c
        
        for j in J:
            
            for l in L:
                
                Sum = 0
                
                for r in self.R:
                    
                    if j==r or l==r:
                        
                        Sum += 0
                        
                    else:
                        
                        for s in self.S:
                            
                            x0=numpy.pi*r**2
                            x1=self.length*j/(-numpy.pi*j**2 + x0)
                            x2=(-1)**r
                            x3=self.length*l/(-numpy.pi*l**2 + x0)
                     
                            Sum += 1j*omega*(2 - self.delta(r))*(2 - self.delta(s))*((-1)**j*x1*x2 - x1)*((-1)**l*x2*x3 - x3)/(self.height*(-k0**2 + 2*1j*k0*self.kappa(r, s)*self.zetars + self.kappa(r, s)**2))
                    
                Zc[j-1,l-1,:] = Sum
                
        return Zc
                            
                            
        
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    