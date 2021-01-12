#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 07:35:22 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from DuctElement import DuctElementDummy, DuctElement
import matplotlib.pyplot as plt
from numpy import log10
from numpy import abs
 
#%%
# =============================================================================
# class Duct(tr.HasTraits):
#     '''
#     Duct combines different DuctElements to build a silencer.
#     '''
#     # height of the duct
#     height = tr.Float()
#     
#     # number of modes
#     M = tr.Int()     # y-direction
#     N = tr.Int()     # z-direction
#     
#     # list with DuctElement objects / transfer matrices
#     elements = tr.List(trait = tr.Instance(DuctElement))
#     
#     def Ij(freq):
#         # defining the excitation
#         pass
#     
#     def tl(self, freq):
#         # TL = [...] calculates TL over all duct elements
#         pass
# =============================================================================
    
    
# =============================================================================
# class Duct2D(tr.HasTraits):
#     '''
#     2D-Duct. Combines different DuctElements to build a silencer.
#     '''
#     
#     # height of the duct
#     height = tr.Float()    
#     
#     M = tr.Int()
#     
#     # list with DuctElement objects / transfer matrices
#     matrices = tr.List()
#     
#     Z = tr.List(tr.CFloat())
#     
#     def impedance(self):
#         return self.Z
#     
#     def tl(self):
#         S0 = 2
#         
#     
#         TL = 20*log10((1/2)*abs(np.array(self.matrices)[0,0,:]+(S0/np.array(self.Z))*np.array(self.matrices)[0,1,:]+(np.array(self.Z)/S0)*np.array(self.matrices)[1,0,:]+np.array(self.matrices)[1,1,:])) 
#         
#         return TL
#     
#     def height(self):
#         return self.height
# =============================================================================

    
class Duct(tr.HasTraits):
    '''
    Combines different DuctElements to build a silencer.
    '''
    # geometrie
    #height = tr.Float()
    
    #
    #freq = tr.List(trait=tr.Int())
    freq = tr.Instance(np.ndarray)
    
    # number of modes
    #M = tr.Int()
    
    # list of duct elements / transfer matrices
    elements = tr.List(trait = tr.Instance(DuctElement))
    
    # TL of reflection and absorption silencer    
    def tl(self):
        
        S0 = 2.0
        Z0 = self.elements[0].medium.c*self.elements[0].medium.rho0
        
        # multiplication of the different element matrice
        #T = self.elements[0].T(self.freq)
  
        for idx, element in enumerate(self.elements):
            print(idx, element)
            T_temp = element.T(self.freq).T
            
            if idx == 0:
                
                T = T_temp
                
            else:
                
                for f in range(len(self.freq)):
                    T[f] = np.dot(T[f], T_temp[f])
                    
        
        # calculation of TL
        T = T.T
        TL = 20*log10((1/2)*abs(T[0,0]+(S0/Z0)*T[0,1]+(Z0/S0)*T[1,0]+T[1,1]))
        
        
        return TL
    
    # transmission loss of plate resonator silencer
    def tlplate(self):
        
        v = self.elements[0].solvelgs(self.freq)
        
        return v
        
    
# =============================================================================
#     def tmatrix(self):
#        
#         for i in self.elements:
#             
#             i.lining.koeff(self.freq)
#             
#             i.tmatrix()
#                 
#     
#     def tl(self):
#         
#         S0 = 2.0
#         Z0 = self.elements[0].medium.c*self.elements[0].medium.rho0
#         
#         # multiplication of the different element matrices
#         for i in range(len(self.elements)):
#             
#             if i==0:
#                 T = self.elements[i].T.T
#                 
#             else:
#                 for f in range(len(self.freq)):
#                     T[f] = np.dot(T[f], self.elements[i].T.T[f])
#         
#         # calculating the TL
#         T = T.T
#         self.TL = 20*log10((1/2)*abs(T[0,0]+(S0/Z0)*T[0,1]+(Z0/S0)*T[1,0]+T[1,1]))
#                     
#         return (self.TL)
# =============================================================================
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      


    

