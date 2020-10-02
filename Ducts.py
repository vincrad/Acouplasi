#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 07:35:22 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from DuctElement import DuctElementDummy2
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
    2D-Duct. Combines different DuctElements to build a silencer.
    '''
    # geometrie
    #height = tr.Float()
    
    # number of modes
    #M = tr.Int()
    
    # list of duct elements / transfer matrices
    elements = tr.List(trait = tr.Instance(DuctElementDummy2))
    
    def tl(self):
        
        S0 = 2.0
        Z0 = 1.2*343
        
        self.TL = 20*log10((1/2)*abs(self.elements[0].T[0,0]+(S0/Z0)*self.elements[0].T[0,1]+(Z0/S0)*self.elements[0].T[1,0]+self.elements[0].T[1,1]))
        
        return self.TL
    
    def test(self, freq):
        
        S0 = 2.0
        Z0 = 1.2*343
        
        # multiplication of the different element matrices
        for i in range(len(self.elements)):
            
            if i==0:
                T = self.elements[i].T.T
                
            else:
                for f in range(len(freq)):
                    T[f] = np.dot(T[f], self.elements[i].T.T[f])
        
        # calculating the TL
        T = T.T
        self.TL = 20*log10((1/2)*abs(T[0,0]+(S0/Z0)*T[0,1]+(Z0/S0)*T[1,0]+T[1,1]))
                    
        return (self.TL)
    
    
    
        
    
    
    
#%% 
# =============================================================================
# amp = 1
# f = 500
# #f = np.arange(0, 501)
# rho = 1,2
# c = 343
# x = np.arange(0, 1.01, 0.01)
# omega = 2*np.pi*f
# 
# k = omega/c
# 
# P = []
# for i in x:
#     p = amp*np.exp(-1j*k*i)
#     P.append(p)
#     
# fig = plt.figure()
# plt.plot(x, P)
# =============================================================================

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      


    

