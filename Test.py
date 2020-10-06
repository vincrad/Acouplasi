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

temp = 20

f = list(range(10,260,1))
#f = list(np.arange(10,260,10))

fluid1 = Fluid()
#c = fluid1.soundspeed(temp)

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


#%% 

# =============================================================================
# f = list(range(10,260,1))
# 
# fluid1 = Fluid()
# 
# lining1 = DummyReflection(length=5, depth=2, height=5, medium=fluid1)
# 
# ductelement1 = DuctElementDummy(lining=lining1, medium=fluid1)
# 
# duct = Duct(elements=[ductelement1], freq=f)
# 
# duct.tmatrix()
# 
# tl = duct.tl()
# 
# fig = plt.plot()
# plt.plot(f, tl)
# =============================================================================










