#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:47:36 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from Fluid import Fluid
from Linings import DummyLining, DummyAbsorption

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

# =============================================================================
# class DuctElementDummy(tr.HasTraits):
#     
#     # Informationen, die eigentlich aus fluid kommen
#     def __init__(self):
#         self.c = 343 #m/s
#         self.rho0 = 1.2  # kg/m³
#         self.S = 2   # m²
#         self.l = 1   # m
#         
#     lining = tr.Instance(ReflectiveSilencer)
#     
# 
#     def tmatrix(self, freq):
#          # Kanal ohne Auskleidung
#          kz = (2*np.pi*freq)/self.c
#         
#          # Schallflussimpedanz
#          Z = ((self.rho0*self.c)/self.S*kz)
#         
#          T = np.array([[np.cos(kz*self.l), 1j*Z*np.sin(kz*self.l)],[1j*(1/Z)*np.sin(kz*self.l), np.cos(kz*self.l)]])
#          
#          return (T, Z)
# =============================================================================

class DuctElementDummy2(tr.HasTraits):
    '''
    Class calculates the transfer matrix of a certain duct section.
    '''
    #flowspeed = tr.Float()
    
    #medium = tr.Instance(Fluid)
    
    lining = tr.Instance(DummyLining)
    
    def tmatrix(self):
        self.T = np.array([[np.cos(self.lining.kz*self.lining.length), 1j*self.lining.Z*np.sin(self.lining.kz*self.lining.length)],[1j*(1/self.lining.Z)*np.sin(self.lining.kz*self.lining.length), np.cos(self.lining.kz*self.lining.length)]])
        
        return self.T
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
        
        
        
        
        
        
        
        