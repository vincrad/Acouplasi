#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:06:20 2020

@author: radmann
"""

##############################################################################
# Schalldämpfer Test
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from DuctElement import DuctElementDummy, DuctElementPlate
from Ducts import Duct
from Fluid import Fluid
from Linings import DummyLining, DummyReflection, DummyAbsorption, PlateResonators, SinglePlate
from Plate import Plate
from Cavity import Cavity



#%% Test 6.1 - Kombination Reflexion/Absorption - Lazy

# =============================================================================
# f = list(range(10,260,1))
# 
# TL = []
# 
# temp = np.arange(20, 430, 200)
# 
# for  i in temp:
# 
#     fluid1 = Fluid(rho0=1.2)
#     c = fluid1.soundspeed(i)
#     
#     lining1 = DummyReflection(length=5, depth=2, height=5, medium=fluid1)
#     
#     lining2 = DummyAbsorption(length=5, depth=2, height=1, dw=0.35, medium=fluid1)
#     
#     ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
#     
#     ductelement2 = DuctElementDummy(lining=lining2, medium=fluid1)
#     
#     duct = Duct(elements=[ductelement1, ductelement2], freq=f)
#     
#     duct.tmatrix()
#     
#     tl = duct.tl()
#     
#     TL.append(tl)
#     
# fig = plt.figure
# for i in TL:
#     plt.plot(f, i)
# =============================================================================



#%% Test 7 - Lazy Evaluation

# =============================================================================
# temp = 20
# 
# f = np.arange(10,260,1)
# #f = list(np.arange(10,260,10))
# 
# fluid1 = Fluid()
# #c = fluid1.soundspeed(temp)
# 
# lining1 = DummyReflection(length=5, depth=2, height=5, medium=fluid1)
# 
# lining2 = DummyReflection(length=1, depth=2, height=1, medium=fluid1)
# 
# lining3 = DummyReflection(length=4, depth=2, height=5, medium=fluid1)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# 
# ductelement2 = DuctElementDummy(lining=lining2, medium=fluid1)
# 
# ductelement3 = DuctElementDummy(lining=lining3, medium=fluid1)
# 
# duct = Duct(elements=[ductelement1, ductelement2, ductelement3], freq=f)
# 
# duct.tmatrix()
# 
# tl2 = duct.tl()
# 
# fig = plt.figure()
# plt.plot(f, tl2)
# =============================================================================

#%%

# =============================================================================
# temp = 20
# 
# f = np.arange(10,260,1)
# 
# fluid1 = Fluid(temperatureC=temp)
# 
# lining1 = DummyReflection(length=5, depth=2, height=5, medium=fluid1)
# 
# lining2 = DummyReflection(length=1, depth=2, height=1, medium=fluid1)
# 
# lining3 = DummyReflection(length=4, depth=2, height=5, medium=fluid1)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# 
# ductelement2 = DuctElementDummy(lining=lining2, medium=fluid1)
# 
# ductelement3 = DuctElementDummy(lining=lining3, medium=fluid1)
# 
# duct = Duct(elements=[ductelement1, ductelement2, ductelement3], freq = f)
# 
# tl = duct.tl()
# 
# fig = plt.figure()
# plt.plot(f, tl)
# =============================================================================


#%% Test - Temperaturabhängigkeit

# =============================================================================
# f = np.arange(10,260,1)
# 
# fluid1 = Fluid()
# 
# lining1 = DummyReflection(length=5, depth=2, height=5, medium=fluid1)
# 
# lining2 = DummyReflection(length=1, depth=2, height=1, medium=fluid1)
# 
# lining3 = DummyReflection(length=4, depth=2, height=5, medium=fluid1)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# 
# ductelement2 = DuctElementDummy(lining=lining2, medium=fluid1)
# 
# ductelement3 = DuctElementDummy(lining=lining3, medium=fluid1)
# 
# duct = Duct(elements=[ductelement1, ductelement2, ductelement3], freq = f)
# 
# 
# temp = np.arange(20.0, 1020.0, 200.0)
# 
# fig = plt.figure()
# for i in temp:
#     
#     fluid1.temperatureC = i
# 
#     tl = duct.tl()
# 
#     plt.plot(f, tl)
# =============================================================================


#%% Test - Temperaturabhängigkeit

# =============================================================================
# f = np.arange(10,260,1)
# 
# fluid1 = Fluid()
# 
# lining1 = DummyReflection(length=5, depth=2, height=5, medium=fluid1)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# 
# T = ductelement1.T
# 
# duct = Duct(elements=[ductelement1], freq=f)
# 
# tl = duct.tl()
# 
# fig = plt.figure()
# plt.plot(f, tl)
# 
# # =============================================================================
# # temp = np.arange(20.0,430.0,200.0)
# # 
# # fluid1.temperatureC = 400
# # 
# # print(lining1.medium.c)
# # 
# # print(ductelement1.lining.medium.c)
# # 
# # print(duct.elements[0].lining.medium.c)
# # 
# # T = duct.elements[0].T
# # #kz = lining1.kz(f)
# # 
# # #tl2 = duct.tl()
# # ===========================================================================
# 
# T2 = ductelement1.T
# =============================================================================


#%% Test plate resonator

# frequency
f = np.arange(10, 260, 10)

# number of plate modes
N = 5

# plate modes
J = np.arange(1, N+1, 1)
L = np.arange(1, N+1, 1)

fluid1 = Fluid()

#cavity = Cavity(hc=5.0)

# cavity modes
R = np.arange(0,5,1)
S = np.arange(0,5,1)

cavity1 = Cavity(length=1, height=0.5, R=R, S=S, medium=fluid1)

lining1 = SinglePlate(length=1, J=J, L=L, cavity=cavity1, medium=fluid1)

ductelement1 = DuctElementPlate(lining=lining1, medium=fluid1)

duct1 = Duct(elements=[ductelement1], freq=f)

Z, I = duct1.tlplate()


















