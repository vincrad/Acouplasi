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
from Linings import PlateResonators, SinglePlateResonator, SimpleTwoSidedPlateResonator
from Plate import Plate, Plate3D
from Cavity import Cavity, Cavity3D

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
r = np.arange(0,10,1)
s = np.arange(0,10,1)

fluid1 = Fluid(temperature=temp1)

material1 = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))

plate1 = Plate(hp=0.0003, material=material1, temperature=temp1)

cavity1 = Cavity(height=1, r=r, s=s, medium=fluid1,)

#lining1 = SinglePlateResonator(length=5, depth=1, j=j, l=l, plate=plate1, cavity=cavity1)
lining1 = SimpleTwoSidedPlateResonator(length=5, depth=1, j=j, l=l, plate=plate1, cavity=cavity1)

ductelement1 = DuctElement(lining=lining1, medium=fluid1, M=0.3)

duct1 = Duct(freq=f, height_d=1, elements=[ductelement1])

Tl = duct1.tl()

plt.figure()
plt.plot(f,Tl)
plt.title('Transmission Loss')
plt.show()

Lmatrix = plate1.lmatrix(5,1,l,f)

#%%

a,r,t = duct1.coefficients()

labels = ['t', 'r', 'a']

fig, ax = plt.subplots()
ax.stackplot(f, t, r, a, labels=labels)
ax.legend()
plt.title('Transmissions-, Reflexions-, Absorptionsgrad')
plt.show()

test = a+r+t

#%% Test plate silencer 3D

# frequency
f = np.arange(10,260,10)

# temperature
temp1 = Temperature(C=20)

# # number of plate modes
N = 5
 
# plate modes
j = np.arange(1, N+1, 1)
k = np.arange(1, N+1, 1)
l = np.arange(1, N+1, 1)
n = np.arange(1, N+1, 1)

# cavity modes
r = np.arange(0,10,1)
s = np.arange(0,10,1)
t = np.arange(0,10,1)

fluid1 = Fluid(temperature=temp1)

material1 = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))

plate2 = Plate3D(hp=0.0003, material=material1, temperature=temp1)

cavity1 = Cavity3D(height=1, r=r, s=s, t=t, medium=fluid1)

Lmatrix_2 = plate2.lmatrix(5,1,l,n,f)

#test = plate1.lmatrix(5,1,l,f)



#%%

test2 = cavity1.cavityimpedance(5,1,j,k,l,n,f)





















