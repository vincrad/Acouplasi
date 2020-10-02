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
from DuctElement import DuctElementDummy
from Ducts import Duct
from Fluid import Fluid
from Linings import DummyLining


#%% Frequenzen
f = np.arange(10, 260, 1)


#%% Test 1 - Kammerschalldämpfer

# =============================================================================
# temp = 20
# 
# fluid1 = Fluid(rho0=1.2)
# c = fluid1.soundspeed(temp)
# 
# lining1 = DummyLining(medium=fluid1, length=5, depth=4, height=5)
# [kz, Z] = lining1.reflection(f)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# T_lining1 = ductelement1.tmatrix()
# 
# duct1 = Duct(elements=[ductelement1])
# tl1 = duct1.tl()
# 
# 
# fig = plt.figure()
# plt.plot(f, tl1)
# plt.axis([10, 250, -1, 15])
# =============================================================================

#%% Test 2 - Absorptionsschalldämpfer

# =============================================================================
# temp = 20
# 
# fluid1 = Fluid(rho0=1.2)
# c = fluid1.soundspeed(temp)
# 
# lining1 = DummyLining(medium=fluid1, length=5, depth=2, height=1, dw=0.35)
# [kz, Z] = lining1.absorption(f)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# T_lining2 = ductelement1.tmatrix()
# 
# 
# duct1 = Duct(elements=[ductelement1])
# tl1 = duct1.tl()
# 
# 
# fig = plt.figure()
# plt.plot(f, tl1)
# plt.axis([10, 250, -1, 15])
# =============================================================================


#%% Test 3 - 2 Reflexionsschalldämpfer

# =============================================================================
# temp = 20
# 
# fluid1 = Fluid(rho0=1.2)
# c = fluid1.soundspeed(temp)
# 
# lining1 = DummyLining(medium=fluid1, length=5, depth=4, height=5)
# [kz1, Z1] = lining1.reflection(f)
# 
# lining2 = DummyLining(medium=fluid1, length=2, depth=4, height=1)
# [kz2, Z2] = lining2.reflection(f)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# T_lining1 = ductelement1.tmatrix()
# 
# ductelement2 = DuctElementDummy(lining=lining2, medium=fluid1)
# T_lining2 = ductelement2.tmatrix()
# 
# duct = Duct(elements=[ductelement1, ductelement2])
# tl1 = duct.tl()
# 
# tl = duct.test(f)
# 
# 
# fig = plt.figure()
# plt.plot(f, tl)
# plt.axis([10, 250, -1, 15])
# =============================================================================


#%% Test 4 - Temperaturabhängigkeit
# =============================================================================
# temp = np.arange(20, 430, 100)
# 
# TL = []
# for i in temp:
#     fluid1 = Fluid(rho0=1.2)
#     c = fluid1.soundspeed(i)
#     
#     lining1 = DummyLining(length=5, depth=4, height=5, medium=fluid1)
#     [kz, Z1] = lining1.reflection(f)
#     
#     ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
#     T_lining1 = ductelement1.tmatrix()
#     
#     duct = Duct(elements=[ductelement1])
#     tl1 = duct.tl()
#     
#     TL.append(tl1)
# 
# fig = plt.figure()
# for i in TL:
#     plt.plot(f, i)
# =============================================================================
    

#%% Test 5 - 3 Reflexionsschalldämpfer

# =============================================================================
# temp = 20
# 
# fluid1 = Fluid(rho0=1.2)
# c = fluid1.soundspeed(temp)
# 
# lining1 = DummyLining(length=5, depth=2, height=5, medium=fluid1)
# [kz1, Z1] = lining1.reflection(f)   
# 
# lining2 = DummyLining(length=1, depth=2, height=1, medium=fluid1)
# [kz2, Z2] = lining2.reflection(f)
# 
# lining3 = DummyLining(length=4, depth=2, height=5, medium=fluid1)
# [kz3, Z3] = lining3.reflection(f)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# T_lining1 = ductelement1.tmatrix()
# 
# ductelement2 = DuctElementDummy(lining=lining2, medium=fluid1)
# T_lining2 = ductelement2.tmatrix()
# 
# ductelement3 = DuctElementDummy(lining=lining3, medium=fluid1)
# T_lining3 = ductelement3.tmatrix()
# 
# duct = Duct(elements=[ductelement1, ductelement2, ductelement3])
# tl = duct.test(f)
# 
# fig = plt.figure
# plt.plot(f, tl)
# =============================================================================

#%% Test 6 - Kombination Reflexion/Absorption

temp = np.arange(20, 430, 200)

TL = []

for i in temp:
    
    fluid1 = Fluid(rho0=1.2)
    c = fluid1.soundspeed(i)
    
    lining1 = DummyLining(length=5, depth=2, height=5, medium=fluid1)
    [kz1, Z1] = lining1.reflection(f)
    
    lining2 = DummyLining(length=5, depth=2, height=1, dw=0.35, medium=fluid1)
    [kz2, Z2] = lining2.absorption(f)
    
    ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
    T_lining1 = ductelement1.tmatrix()
    
    ductelement2 = DuctElementDummy(lining=lining2, medium=fluid1)
    T_lining2 = ductelement2.tmatrix()
    
    duct = Duct(elements=[ductelement1, ductelement2])
    tl = duct.test(f)
    
    TL.append(tl)

fig = plt.figure()
for i in TL:
    
    plt.plot(f, i)



























