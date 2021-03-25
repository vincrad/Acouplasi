#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:06:20 2020

@author: radmann
"""

##############################################################################
# Schalldämpfer Test
##############################################################################

import numpy as np, numpy
import matplotlib.pyplot as plt
from DuctElement import DuctElement
from Ducts import Duct
from Temperature import Temperature
from Fluid import Fluid
from Material import Material
from Linings import PlateResonators, SinglePlateResonator
from Plate import Plate
from Cavity import Cavity

#%% Test plate resonator 

# frequency
f = np.arange(10,260,10)

# temperature
temp1 = Temperature(C=20)

# # number of plate modes
N = 5
 
# plate modes
j = np.arange(1, N+1, 1)
l = np.arange(1, N+1, 1)

# cavity modes
r = np.arange(0,5,1)
s = np.arange(0,5,1)

fluid1 = Fluid(temperature=temp1)

material1 = Material(rhop=3500, temperature=temp1)

plate1 = Plate(hp=0.01, material=material1)

cavity1 = Cavity(height=1, r=r, s=s, medium=fluid1)

lining1 = SinglePlateResonator(length=1, depth=1, j=j, l=l, plate=plate1, cavity=cavity1)

ductelement1 = DuctElement(lining=lining1, medium=fluid1, M=0)

duct1 = Duct(freq=f, height_d=1, elements=[ductelement1])

Tl = duct1.tl()


# impedance matrizen

Zc = cavity1.cavityimpedance(1, 1, j, l, f)

Z = lining1.zmatrix(1, 0, f)









