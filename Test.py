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
from Linings import DummyLining, DummyReflection, DummyAbsorption


#%% Frequenzen
#f = np.arange(10, 260, 1)


#%% Test 1 - Kammerschalldämpfer

# =============================================================================
# temp = 20
# 
# fluid1 = Fluid(rho0=1.2)
# c = fluid1.soundspeed(temp)
# 
# lining1 = DummyReflection(medium=fluid1, length=5, depth=4, height=5)
# [kz, Z] = lining1.reflection(f)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# T_lining1 = ductelement1.tmatrix()
# 
# duct1 = Duct(elements=[ductelement1])
# tl1 = duct1.tl(f)
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
# lining1 = DummyAbsorption(medium=fluid1, length=5, depth=2, height=1, dw=0.35)
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
# lining1 = DummyReflection(medium=fluid1, length=5, depth=4, height=5)
# [kz1, Z1] = lining1.reflection(f)
# 
# lining2 = DummyReflection(medium=fluid1, length=2, depth=4, height=1)
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
#     lining1 = DummyReflection(length=5, depth=4, height=5, medium=fluid1)
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
# lining1 = DummyReflection(length=5, depth=2, height=5, medium=fluid1)
# [kz1, Z1] = lining1.reflection(f)   
# 
# lining2 = DummyReflection(length=1, depth=2, height=1, medium=fluid1)
# [kz2, Z2] = lining2.reflection(f)
# 
# lining3 = DummyReflection(length=4, depth=2, height=5, medium=fluid1)
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

# =============================================================================
# temp = np.arange(20, 430, 200)
# 
# TL = []
# 
# for i in temp:
#     
#     fluid1 = Fluid(rho0=1.2)
#     c = fluid1.soundspeed(i)
#     
#     lining1 = DummyReflection(length=5, depth=2, height=5, medium=fluid1)
#     [kz1, Z1] = lining1.reflection(f)
#     
#     lining2 = DummyAbsorption(length=5, depth=2, height=1, dw=0.35, medium=fluid1)
#     [kz2, Z2] = lining2.absorption(f)
#     
#     ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
#     T_lining1 = ductelement1.tmatrix()
#     
#     ductelement2 = DuctElementDummy(lining=lining2, medium=fluid1)
#     T_lining2 = ductelement2.tmatrix()
#     
#     duct = Duct(elements=[ductelement1, ductelement2])
#     tl = duct.tl(f)
#     
#     TL.append(tl)
# 
# fig = plt.figure()
# for i in TL:
#     
#     plt.plot(f, i)
# =============================================================================



#%% Test 7 - Lazy Evaluation

temp = 20

f = list(range(10,260,1))
#f = list(np.arange(10,260,10))

fluid1 = Fluid(rho0=1.2)
c = fluid1.soundspeed(temp)

lining1 = DummyReflection(length=5, depth=2, height=5, medium=fluid1)

lining2 = DummyReflection(length=1, depth=2, height=1, medium=fluid1)

lining3 = DummyReflection(length=4, depth=2, height=5, medium=fluid1)

ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)

ductelement2 = DuctElementDummy(lining=lining2, medium=fluid1)

ductelement3 = DuctElementDummy(lining=lining3, medium=fluid1)

duct = Duct(elements=[ductelement1, ductelement2, ductelement3], freq=f)

duct.tmatrix()

tl2 = duct.tl()

fig = plt.figure()
plt.plot(f, tl2)



# =============================================================================
# elements=[ductelement1]
# 
# for i in elements:
#     print(i)
#     print(i.lining)
#     print(i.lining.length)
#     print(i.lining.medium.c)
#     
#     [kz, Z] = i.lining.reflection(f)
# =============================================================================
















