#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:47:36 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
from Fluid import Fluid
from Linings import DummyLining, DummyReflection, DummyAbsorption, PlateResonators, SinglePlate

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

class DuctElement(tr.HasTraits):
    
    '''
    Parental class of different duct elements.
    '''
    #flowspeed = tr.Float()
    
    medium = tr.Instance(Fluid)
    

class DuctElementDummy(DuctElement):
    '''
    Class calculates the transfer matrix of a certain duct section.
    '''
    
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
        
class DuctElementPlate(DuctElement):
    
    '''
    Duct element class for plate resonator linings.
    '''
    
    lining = tr.Instance(PlateResonators)
    
    def incidentsound(self, freq):
        
        omega = 2*np.pi*freq
        
        k0 = omega/self.medium.c
        
        I = np.zeros((len(self.lining.L), len(omega)), dtype=complex)


        for l in self.lining.L:
    
            x0=self.lining.length**2*k0**2
            x1=numpy.pi**2*l**2
            x2=numpy.pi*self.lining.length*l
            x3=numpy.exp(1j*self.lining.length*k0)
    
            I[l-1,:] = (-1)**l*x2/(x0*x3 - x1*x3) - x2/(x0 - x1)
            
        return I
        
    
    def solvelgs(self, freq):
        
        # impedance matrix of plate resonator liner
        Z = self.lining.zmatrix(freq)
        
        # L-Matrix with material properties of the plate
        
        # incident sound
        I = self.incidentsound(freq)
        
        # return Plattenschnelle
        return [Z, I]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
        
        
        
        
        
        
        
        