#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:49:08 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
from Fluid import Fluid

#%%

class Cavity(tr.HasTraits):
    
    '''
    Class to define a cavity for a plate resonator silencer.
    '''
    
    # height of the cavity
    height = tr.Float()
    
    # modes of the cavity
    r = tr.Instance(np.ndarray)
    s = tr.Instance(np.ndarray)
    
    # loss factor
    zetars = tr.Float(default_value=0.1)
    
    # medium
    medium = tr.Instance(Fluid)   
    
    # property methods to define expanded arrays of modes
    @property
    def R(self):
        
        return self.r[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        
    
    @property
    def S(self):
        
        return self.s[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
       
    
    # property methods to define the Kronecker delta
    @property
    def deltar(self):    
        
        return np.eye(len(self.r),1)[np.newaxis, np.newaxis, :, np.newaxis]
    
    
    @property
    def deltas(self):
        
        return np.eye(len(self.s),1)[np.newaxis, np.newaxis, np.newaxis, :]
        
    
    # method to calculate kappars
    def kappars(self, length):
        
        return np.sqrt((self.R*np.pi/length)**2+(self.S*np.pi/self.height)**2)
        
        
    
    # method calculate the cavity impedance matrix
    def cavityimpedance(self, length, depth, j, l, freq):
        
        # define expanded arrays of modes
        L = l[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        J = j[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c
        
        # calculate the cavity impedance matrix
        x0=numpy.pi*self.R**2
        #x1=length*J/(-numpy.pi*J**2 + x0)
        x1 = np.divide(length*J, (-numpy.pi*J**2 + x0), out=np.zeros_like(J*self.R, dtype=float), where=(-numpy.pi*J**2 + x0)!=0)
        x2=(-1)**self.R
        #x3=length*L/(-numpy.pi*L**2 + x0)
        x3 = np.divide(length*L, (-numpy.pi*L**2 + x0), out=np.zeros_like(L*self.R, dtype=float), where=(-numpy.pi*L**2 + x0)!=0)
        
        Zc_temp = 1j*omega*self.medium.rho0*(2 - self.deltar)*(2 - self.deltas)*((-1)**J*x1*x2 - x1)*((-1)**L*x2*x3 - x3)/(length*self.height*(-k0**2 + 2*1j*k0*self.kappars(length)*self.zetars + self.kappars(length)**2))
        
        # building the final Zc matrix by summation over R and S
        Zc = np.sum(Zc_temp, axis=(2,3))
        
        return Zc
        
        
        
                         
                            
        
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    