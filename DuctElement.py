#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:47:36 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from Fluid import Fluid
from Linings import DummyLining, DummyReflection, DummyAbsorption

# =============================================================================
# class DuctElement(tr.HasTraits):
#     '''
#     Class calculates the transfer matrix of a certain duct section.
#     '''
#     flowspeed = tr.Float()
#     
#     fluid = tr.Instance(Fluid) 
#     
#     lining = tr.Instance(PlateResonator)
#     
#     def tmatrix(self, freq):
#         # function calculates the transfer matrix of the duct section
#         pass
#     
#     # Anregung wird gebraucht und Z- und L-Matrix, um GS zu lösen
#     # daraus Berechnung der Transfermatrix
# =============================================================================


class DuctElementDummy(tr.HasTraits):
    '''
    Class calculates the transfer matrix of a certain duct section.
    '''
    #flowspeed = tr.Float()
    
    medium = tr.Instance(Fluid)
    
    lining = tr.Instance(DummyLining)
    
    @property
    def depth(self):
        
        depth = self.lining.depth
        
        return depth
    
    
    def T(self,freq):
        
        [kz, Z] = self.lining.Zkz(freq)
        
# =============================================================================
#         kz = self.lining.kz(freq)
#         Z = self.lining.Z()
# =============================================================================
        
        T = np.array([[np.cos(kz*self.lining.length), 1j*Z*np.sin(kz*self.lining.length)],[1j*(1/Z)*np.sin(kz*self.lining.length), np.cos(kz*self.lining.length)]])
        
        return T

        
    
    
    
# =============================================================================
#     def tmatrix(self):
#         
#         # wie übergibt man das am schlausten?
#         self.depth = self.lining.depth
#     
#         self.T = np.array([[np.cos(self.lining.kz*self.lining.length), 1j*self.lining.Z*np.sin(self.lining.kz*self.lining.length)],[1j*(1/self.lining.Z)*np.sin(self.lining.kz*self.lining.length), np.cos(self.lining.kz*self.lining.length)]])
#         
#         return self.T
# =============================================================================
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
        
        
        
        
        
        
        
        