#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:08:32 2021

@author: radmann
"""

#%% Mockup-Test

import numpy as np
from Mockup_Code import Fluid, Material, Plate, Cavity, SinglePlateResonator, DuctElement, Duct, ReflectionLining, AbsorptionLining

#%%
# frequency
f = np.arange(10, 260, 10)

# number of modes
N = 5

# plate modes
J = np.arange(1, N+1, 1)
L = np.arange(1, N+1, 1)

# cavity modes
R = np.arange(0, 5, 1)
S = np.arange(0, 5, 1)


fluid1 = Fluid()

material1 = Material(rhop=35)

plate1 = Plate(hp=0.001, material=material1)

cavity1 = Cavity(height=2, R=R, S=S, medium=fluid1)

lining1 = SinglePlateResonator(length=3, J=J, L=L, plate=plate1, cavity=cavity1)

ductelement1 = DuctElement(lining=lining1, medium=fluid1)

duct1 = Duct(elements=[ductelement1], freq=f)

transmissionloss = duct1.tl()

