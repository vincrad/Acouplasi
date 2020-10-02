#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:58:19 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from Fluid import Fluid

# =============================================================================
# class PlateResonator(tr.HasTraits):
#     '''
#     Parent class for all types of plate resonators linings.
#     '''
#     # geometrie
#     length = tr.Float()
#     depth = tr.Float()
#     
#     # number of modes of the plate
#     Lp = tr.Int()   # x-direction
#     Mp = tr.Int()   # y-direction
#     
#     # number of modes of the cavity
#     Lc = tr.Int()   # x-direction
#     Mc = tr.Int()   # y-direction
#     Nc = tr.Int()   # z-direction
#     
#     # transfer of configuration, e.g. StandardPlateResonator 
#     arrangement = tr.Instance()
# =============================================================================
    
    
# =============================================================================
# class ReflectiveSilencer(tr.HasTraits):
#     '''
#     Class for reflective silencer.
#     '''
#     
#     # geometrie
#     length = tr.Float()
#     depth = tr.Float()
#     height = tr.Float()
#     
#     flu = tr.Instance(Fluid)
#     
#     def speed(self):
#         print(self.flu.c)
#         self.c = self.flu.c
#         
#     def 
#     
#     #c = fluid.soundspeed()
#     #Sk = depth*height
#           
#     
# # =============================================================================
# #     def __init__(self, freq):
# #         self.Sk = self.depth*self.height
# #         self.c = Fluid.soundspeed()
# #         self.kz = (2*np.pi*freq)/self.c
# # =============================================================================
#     
# =============================================================================

class DummyLining(tr.HasTraits):
    
    # geometrie
    length = tr.Float()
    depth = tr.Float()
    
    S = tr.Float()
    
    medium = tr.Instance(Fluid)
    
# =============================================================================
#     # random
#     kz = tr.CFloat()
#     Z = tr.CFloat()
# =============================================================================

    # schallharte Wand  / Reflexionsschalldämpfer
    def koeff(self, freq):
        self.kz = (2*np.pi*freq)/self.medium.c
        self.Z = (self.medium.rho0*self.medium.c*self.kz)/(self.S*self.kz)
        
        return (self.kz, self.Z)
        
    
class DummyAbsorption(tr.HasTraits):
    
    # geometrie
    length = tr.Float()
    depth = tr.Float()
    height = tr.Float()
    dw = tr.Float()
    
    # flow resistance
    Xi = tr.Float(default_value=6000)   # Ns/m⁴
    
    
    
    medium = tr.Instance(Fluid)
    
    def koeff(self, freq):
        
        self.S = self.depth*self.height
        
        X = (self.medium.rho0*freq)/self.Xi
        k = (2*np.pi*freq)/self.medium.c
        Z0 = self.medium.rho0*self.medium.c
        
        # nach Miki
        ka = (1+0.0109*X**(-0.618)-1j*0.160*X**(-0.618))*k
        Za = (1+0.070*X**(-0.632)-1j*0.107*X**(-0.632))*Z0
        
        self.Zw = -1j*Za*(1/np.tan(ka*self.dw))
        
        
        self.kz = np.sqrt(k**2-((1j*Z0)/(self.Zw*self.height)))
        self.Z = (Z0*k)/(self.S*self.kz)
        
        return (self.kz, self.Z)
        
#%%
# =============================================================================
# f = np.arange(1, 200, 11)
# F = Fluid()
# test = F.c
# 
# silencer = ReflectiveSilencer(length=4, depth=1, height=2)
# silencer.flu = F
# print(silencer.length)
# print(silencer.depth)
# print(silencer.flu)
# print(silencer.flu.c)
# 
# silencer.speed()
# 
# print(silencer.c)
# 
# #print(silencer.fluid.c)
# 
# 
# #print(silencer.Sk)
# =============================================================================





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    