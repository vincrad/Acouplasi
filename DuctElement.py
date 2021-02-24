#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:47:36 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
from Fluid import Fluid
from Linings import Linings, PlateResonators, SinglePlateResonator

# =============================================================================
# # =============================================================================
# # class DuctElement(tr.HasTraits):
# #     '''
# #     Class calculates the transfer matrix of a certain duct section.
# #     '''
# #     flowspeed = tr.Float()
# #     
# #     fluid = tr.Instance(Fluid) 
# #     
# #     lining = tr.Instance(PlateResonator)
# #     
# #     def tmatrix(self, freq):
# #         # function calculates the transfer matrix of the duct section
# #         pass
# #     
# #     # Anregung wird gebraucht und Z- und L-Matrix, um GS zu lösen
# #     # daraus Berechnung der Transfermatrix
# # =============================================================================
# 
# class DuctElement(tr.HasTraits):
#     
#     '''
#     Parental class of different duct elements.
#     '''
#     #flowspeed = tr.Float()
#     
#     medium = tr.Instance(Fluid)
#     
# 
# # =============================================================================
# # class DuctElementDummy(DuctElement):
# #     '''
# #     Class calculates the transfer matrix of a certain duct section.
# #     '''
# #     
# #     lining = tr.Instance(DummyLining)
# #     
# #     @property
# #     def depth(self):
# #         
# #         depth = self.lining.depth
# #         
# #         return depth
# #     
# #     
# #     def T(self,freq):
# #         
# #         [kz, Z] = self.lining.Zkz(freq)
# #         
# # # =============================================================================
# # #         kz = self.lining.kz(freq)
# # #         Z = self.lining.Z()
# # # =============================================================================
# #         
# #         T = np.array([[np.cos(kz*self.lining.length), 1j*Z*np.sin(kz*self.lining.length)],[1j*(1/Z)*np.sin(kz*self.lining.length), np.cos(kz*self.lining.length)]])
# #         
# #         return T
# # =============================================================================
# 
# 
# # =============================================================================
# #     def tmatrix(self):
# #         
# #         # wie übergibt man das am schlausten?
# #         self.depth = self.lining.depth
# #     
# #         self.T = np.array([[np.cos(self.lining.kz*self.lining.length), 1j*self.lining.Z*np.sin(self.lining.kz*self.lining.length)],[1j*(1/self.lining.Z)*np.sin(self.lining.kz*self.lining.length), np.cos(self.lining.kz*self.lining.length)]])
# #         
# #         return self.T
# # =============================================================================
#         
# class DuctElementPlate(DuctElement):
#     
#     '''
#     Duct element class for plate resonator linings.
#     '''
#     
#     lining = tr.Instance(PlateResonators)
#     
#     def incidentsound(self, freq):
#         
#         omega = 2*np.pi*freq
#         
#         k0 = omega/self.medium.c
#         
#         I = np.zeros((len(self.lining.L), len(omega)), dtype=complex)
# 
# 
#         for l in self.lining.L:
#     
#             x0=self.lining.length**2*k0**2
#             x1=numpy.pi**2*l**2
#             x2=numpy.pi*self.lining.length*l
#             x3=numpy.exp(1j*self.lining.length*k0)
#     
#             I[l-1,:] = (-1)**l*x2/(x0*x3 - x1*x3) - x2/(x0 - x1)
#             
#         return I
#         
#     
#     def solvelgs(self, freq):
#         
#         # impedance matrix of plate resonator liner
#         Z = self.lining.zmatrix(freq)
#         
#         # L-Matrix with material properties of the plate
#         Lmatrix = self.lining.plate.lmatrix(self.lining.length, self.lining.depth, self.lining.J, self.lining.L, freq)
#         
#         # incident sound
#         I = self.incidentsound(freq)
#         
#         # solving the linear system of equation
#         
#         # building lhs matrix from Z and Lmatrix
#         lhs = Z+Lmatrix            
#         
#         # plate velocity array
#         vp = np.zeros((len(self.lining.L), len(freq)), dtype=complex)
# 
#         for idx in range(len(freq)):
#             
#             vp[:,idx] = np.linalg.solve(lhs[:,:,idx], I[:,idx])
# 
#         
#         # return Plattenschnelle
#         return [vp, lhs, Z, Lmatrix, I]
# =============================================================================

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
        
        x0=2*M
        x1=M**2
        x2=1j*self.lining.length*k0/(M + 1)
        x3=numpy.pi**2*L**2
        
        I = numpy.pi*L*((-1)**(L + 1) + numpy.exp(x2))*(x0 + x1 + 1)*numpy.exp(-x2)/(-self.lining.length**2*k0**2 + x0*x3 + x1*x3 + x3)
        
        return I
    
    # method calculates the transfer matrix for reflection and absorption silencer
    # or returns the transmission loss for plate silencer
    def tmatrix(self, freq):
        
        if isinstance(self.lining, PlateResonators)==True:
            
            I = self.incidentsound(self.M, freq)
            
            TL = self.lining.transmissionloss(I, self.M, self.medium, freq)
            
            return TL
        
        else:
            
            def T(self, freq):
                
                [kz, Z] = self.lining.Zkz(self.medium, freq)
   
                T = np.array([[np.cos(kz*self.lining.length), 1j*Z*np.sin(kz*self.lining.length)],[1j*(1/Z)*np.sin(kz*self.lining.length), np.cos(kz*self.lining.length)]])
        
            return T
    
    
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

        
        
        
        
        
        
        
        
        