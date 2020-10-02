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
from DuctElement import DuctElementDummy2
from Ducts import Duct
from Fluid import Fluid
from Linings import DummyLining, DummyAbsorption

#%%

# =============================================================================
# # mit f=0 testen!
# f = np.arange(0.1, 201, 20)
# 
# ductelement = DuctElementDummy()
# 
# [T, Z] = ductelement.tmatrix(f)
# 
# T_list = T.tolist()
# 
# Z_list = Z.tolist()
# 
# duct1 = Duct2D(height=1.0, M=0, matrices=T_list, Z = Z_list)
# 
# test = duct1.impedance()
# 
# TL = duct1.tl()
# 
# 
# 
# fig = plt.figure()
# plt.plot(f, TL)
# =============================================================================

#%% Test2

# =============================================================================
# f = np.arange(0, 250, 10)
# 
# lining1 = DummyLining(kz=5, Z=7.5, length=5)
# 
# ductelement1 = DuctElementDummy2(lining=lining1)
# 
# T_lining1 = ductelement1.tmatrix()
# print(ductelement1.lining)
# 
# duct1 = Duct2D(elements=[ductelement1])
# tl1 = duct1.tl()
# =============================================================================

#%% Frequenzen
f = np.arange(10, 260, 1)


#%% Test 3 - Kammerschalldämpfer

# =============================================================================
# fluid1 = Fluid(c=343, rho0=1.2)
# 
# lining1 = DummyLining(medium=fluid1, length=5, depth=0, S=20)
# [kz, Z] = lining1.koeff(f)
# 
# ductelement1 = DuctElementDummy2(lining=lining1)
# T_lining1 = ductelement1.tmatrix()
# 
# duct1 = Duct2D(elements=[ductelement1])
# tl1 = duct1.tl()
# 
# 
# fig = plt.figure()
# plt.plot(f, tl1)
# plt.axis([10, 250, -1, 15])
# =============================================================================

#%% Test 4 - Absorptionsschalldämpfer

# =============================================================================
# fluid2 = Fluid(c=343, rho0=1.2)
# 
# lining2 = DummyAbsorption(medium=fluid2, length=5, depth=2, height=1, dw=0.35)
# [kz, Z] = lining2.koeff(f)
# 
# ductelement2 = DuctElementDummy2(lining=lining2)
# T_lining2 = ductelement2.tmatrix()
# 
# 
# duct2 = Duct2D(elements=[ductelement2])
# tl2 = duct2.tl()
# 
# 
# fig = plt.figure()
# plt.plot(f, tl2)
# plt.axis([10, 250, -1, 15])
# =============================================================================


#%% Test 5 - 2 Reflexionsschalldämpfer
# =============================================================================
# fluid1 = Fluid(c=343, rho0=1.2)
# 
# lining1 = DummyLining(medium=fluid1, length=5, depth=0, S=20)
# [kz1, Z1] = lining1.koeff(f)
# 
# lining2 = DummyLining(medium=fluid1, length=2, depth=0, S=5)
# [kz2, Z2] = lining2.koeff(f)
# 
# ductelement1 = DuctElementDummy2(lining=lining1)
# T_lining1 = ductelement1.tmatrix()
# 
# ductelement2 = DuctElementDummy2(lining=lining2)
# T_lining2 = ductelement2.tmatrix()
# 
# duct = Duct2D(elements=[ductelement1, ductelement2])
# tl1 = duct.tl()
# 
# tl = duct.test(f)
# 
# 
# fig = plt.figure()
# plt.plot(f, tl)
# plt.axis([10, 250, -1, 15])
# 
# =============================================================================

#%% Test 6 - Temperaturabhängigkeit
temp = np.arange(20, 430, 100)

TL = []
for i in temp:
    fluid1 = Fluid(rho0=1.2)
    c = fluid1.soundspeed(i)
    
    lining1 = DummyLining(medium=fluid1, length=5, depth=0, S=20)
    [kz, Z1] = lining1.koeff(f)
    
    ductelement1 = DuctElementDummy2(lining=lining1)
    T_lining1 = ductelement1.tmatrix()
    
    duct = Duct(elements=[ductelement1])
    tl1 = duct.tl()
    
    TL.append(tl1)

fig = plt.figure()
for i in TL:
    plt.plot(f, i)
    




























