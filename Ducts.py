#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 07:35:22 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from DuctElement import DuctElement, DuctElement3D
import matplotlib.pyplot as plt
from numpy import log10
from numpy import abs
 
#%%
       
class Duct(tr.HasTraits):

    '''
    Combines different DuctElements to build a silencer.
    '''
    
    # frequency
    freq = tr.Instance(np.ndarray)
    
    # duct height
    height_d = tr.Float()

    # list of DuctElements
    elements = tr.List(trait = tr.Instance(DuctElement))
    
    # method calculate teh transmission loss of the duct
    def tl(self):
        
        for i in self.elements:
            
            TL = i.tmatrix(self.height_d, self.freq)
            
            return TL
        
    # method calculates the transmission loss of the duct from transfer matrices
    def tl2(self):
        
        TM = np.empty((2,2,len(self.freq)), dtype=complex)
        
        for idx1, item in enumerate(self.elements):
            
            tm = item.tmatrix2(self.height_d, self.freq)
            
            if idx1 == 0:
                
                TM = tm
            
            else:
                
                for idx2, f in enumerate(self.freq):
                    
                    TM[:,:,idx2] = TM[:,:,idx2] @ tm[:,:,idx2]
            
            
        # characteristic flow impedance of the 1st and n-th element
        Z1 = self.elements[0].medium.rho0*self.elements[0].medium.c/self.height_d
        Zn = self.elements[-1].medium.rho0*self.elements[-1].medium.c/self.height_d
        
        Z0 = self.elements[0].medium.rho0*self.elements[0].medium.c
        
        # Mach number of the 1st and n-th element
        M1 = self.elements[0].M
        Mn = self.elements[-1].M
        
        # Transmission loss of the whole duct - Mechel/Munjal
        TL = 20*np.log10((Zn/Z1)**(1/2)*((1+M1)/(2*(1+Mn)))*np.abs(TM[0,0,:]+TM[0,1,:]/Zn+TM[1,0,:]*Z1+(Z1/Zn)*TM[1,1,:]))
        
        return TL
        
    # method calculate the coefficients of the duct
    def coefficients(self):
        
        for i in self.elements:
            
            tra, ref, dis = i.coefficients(self.height_d, self.freq)
            
            return tra, ref, dis
    
    
#%%
class Duct3D(tr.HasTraits):

    '''
    Combines different DuctElements to build a silencer.
    '''
    
    # frequency
    freq = tr.Instance(np.ndarray)
    
    # duct height
    height_d = tr.Float()

    # list of DuctElements
    elements = tr.List(trait = tr.Instance(DuctElement3D))
    
    # method calculate teh transmission loss of the duct
    def tl(self):
        
        for i in self.elements:
            
            TL = i.tmatrix(self.height_d, self.freq)
            
            return TL
        
    # method calculate the coefficients of the duct
    def coefficients(self):
        
        for i in self.elements:
            
            alpha, beta, tau = i.coefficients(self.height_d, self.freq)
            
            return alpha, beta, tau
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      


    

