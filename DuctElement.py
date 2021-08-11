#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:47:36 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
from Fluid import Fluid
from Linings import Linings, PlateResonators, SinglePlateResonator, SinglePlateResonator3D

#%%

class DuctElement(tr.HasTraits):
    
    '''
    Class calculates the transfer matrix of a certain duct section.
    '''
    
    # lining of the duct element
    lining = tr.Instance(Linings)
    
    # fluid
    medium = tr.Instance(Fluid)
    
    # flow
    M = tr.Float(default_value=0)
    
    # method calculate the incident sound array for plate silencer
    def incidentsound(self, M, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c
        
        # define expended array of modes
        L = self.lining.l[:, np.newaxis]
        
        # calculate incident sound array
        x0=self.lining.length**2*k0**2
        x1=numpy.pi**2*L**2
        x2=2*M
        x3=M**2
        x4=x1*x3
        x5=numpy.pi*self.lining.length*L
        x6=x5/(x0 - x1*x2 - x1 - x4)
        x7=numpy.exp(1j*self.lining.length*k0/(M + 1))
        x8=x1*x7
        x9=(-1)**L*x5/(x0*x7 - x2*x8 - x4*x7 - x8)
        
        I = self.medium.c**2*self.medium.rho0*(-x2*x6 + x2*x9 - x3*x6 + x3*x9 - x6 + x9)
        
        return I
    
    # method calculates the transfer matrix for reflection and absorption silencer
    # or returns the transmission loss for plate silencer
    def tmatrix(self, height_d, freq):
        
        if isinstance(self.lining, PlateResonators)==True:
            
            # incident sound
            I = self.incidentsound(self.M, freq)
            
            # plate velocity
            vp = self.lining.platevelocity(height_d, I, self.M, freq)
            
            TL = self.lining.transmissionloss(vp, height_d, I, self.M, self.medium, freq)
            
            return TL
        
        else:
            
            def T(self, freq):
                
                [kz, Z] = self.lining.Zkz(self.medium, freq)
   
                T = np.array([[np.cos(kz*self.lining.length), 1j*Z*np.sin(kz*self.lining.length)],[1j*(1/Z)*np.sin(kz*self.lining.length), np.cos(kz*self.lining.length)]])
        
            return T
    
    # method calculate the transmission, reflection and absorption coefficient of the plate silencer duct element
    def coefficients(self, height_d, freq):
        
        # incident sound
        I = self.incidentsound(self.M, freq)
        
        # plate velocity
        vp = self.lining.platevelocity(height_d, I, self.M, freq)
        
        # transmission coefficient
        tau = self.lining.transmission(vp, height_d, I, self.M, self.medium, freq)
        
        # reflection coefficient
        beta = self.lining.reflection(vp, height_d, I, self.M, self.medium, freq)
        
        # absorption coefficient
        alpha = self.lining.absorption(vp, height_d, I, self.M, self.medium, freq)
        
        return alpha, beta, tau
        
#%%

class DuctElement3D(tr.HasTraits):
    
    '''
    Class calculates the transfer matrix of a certain 3D duct section.
    '''
    
    # lining of the duct element
    lining = tr.Instance(Linings) # anpassen 2D/3D
    
    # fluid
    medium = tr.Instance(Fluid)
    
    # flow
    M = tr.Float(default_value=0)
    
    # method calculate the incident sound array for plate silencer
    def incidentsound(self, M, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c

        # define expended arrays of modes
        L = self.lining.l[:, np.newaxis, np.newaxis]
        N = self.lining.n[np.newaxis, :, np.newaxis]
        
        # calculate incident sound array
        x0=2*M
        x1=(M)**(2+0j)
        x2=1j*self.lining.length*k0/(M + 1)
        x3=(np.pi)**(2+0j)*(L)**(2+0j)
        
        I = -self.lining.length*self.lining.depth*(self.medium.c)**(2+0j)*L*self.medium.rho0*((-1)**(N+0j) - 1)*((-1)**(L + 1+0j) + numpy.exp(x2))*(x0 + x1 + 1)*numpy.exp(-x2)/(N*(-(self.lining.length)**(2+0j)*(k0)**(2+0j) + x0*x3 + x1*x3 + x3))
        
        return I

    # method returns the transmission loss for plate silencer
    def tmatrix(self, height_d, freq):
        
        if isinstance(self.lining, PlateResonators)==True:
            
            # incident sound
            I = self.incidentsound(self.M, freq)
            
            # plate velocity
            vp = self.lining.platevelocity(height_d, I, self.M, freq)
            
            TL = self.lining.transmissionloss(vp, height_d, I, self.M, self.medium, freq)
            
            return TL
        
        else:
            
            print('Wrong dimensions of plate silencer!')
            
    # method calculate the transmission, reflection and absorption coefficient of the plate silencer duct element
    def coefficients(self, height_d, freq):
        
        # incident sound
        I = self.incidentsound(self.M, freq)
        
        # plate velocity
        vp = self.lining.platevelocity(height_d, I, self.M, freq)
        
        # transmission coefficient
        tau = self.lining.transmission(vp, height_d, I, self.M, self.medium, freq)
        
        # reflection coefficient
        beta = self.lining.reflection(vp, height_d, I, self.M, self.medium, freq)
        
        # absorption coefficient
        alpha = self.lining.absorption(vp, height_d, I, self.M, self.medium, freq)
        
        return alpha, beta, tau
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
        
        
        
        
        
        
        
        