#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:06:20 2020

@author: radmann
"""

#%%

##############################################################################
# Schalldämpfer Test
##############################################################################


import numpy as np, numpy
import matplotlib.pyplot as plt
from DuctElement import DuctElement, DuctElement3D
from Ducts import Duct, Duct3D
from Temperature import Temperature
from Fluid import Fluid
from Material import Material
from Linings import PlateResonators, SinglePlateResonator, SimpleTwoSidedPlateResonator, SinglePlateResonator3D, SimpleTwoSidedPlateResonator3D
from Plate import DoubleLayerPlate3D, Plate, Plate3D, SimplePlate3D, TripleLayerPlate3D
from Cavity import Cavity, Cavity3D

import time
import numba as nb

#%% Test plate resonator 

# =============================================================================
# # frequency
# f = np.arange(10,260,10)

# # temperature
# temp1 = Temperature(C=20)

# # # number of plate modes
# N = 5
 
# # plate modes
# j = np.arange(1, N+1, 1)
# l = np.arange(1, N+1, 1)

# # cavity modes
# r = np.arange(0,10,1)
# s = np.arange(0,10,1)

# fluid1 = Fluid(temperature=temp1)

# material1 = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))

# plate1 = Plate(hp=0.0003, material=material1, temperature=temp1)

# cavity1 = Cavity(height=1, r=r, s=s, medium=fluid1,)

# #lining1 = SinglePlateResonator(length=5, depth=1, j=j, l=l, plate=plate1, cavity=cavity1)
# lining1 = SimpleTwoSidedPlateResonator(length=5, depth=1, j=j, l=l, plate=plate1, cavity=cavity1)

# ductelement1 = DuctElement(lining=lining1, medium=fluid1, M=0.3)

# duct1 = Duct(freq=f, height_d=1, elements=[ductelement1])

# Tl = duct1.tl()

# plt.figure()
# plt.plot(f,Tl)
# plt.title('Transmission Loss')
# plt.show()

# Lmatrix = plate1.lmatrix(5,1,l,f)

# #%%

# a,r,t = duct1.coefficients()

# labels = ['t', 'r', 'a']

# fig, ax = plt.subplots()
# ax.stackplot(f, t, r, a, labels=labels)
# ax.legend()
# plt.title('Transmissions-, Reflexions-, Absorptionsgrad')
# plt.show()

# test = a+r+t
# =============================================================================

#%% Test plate silencer 3D

# frequency
f = np.arange(10, 260, 10)

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

plate1 = SimplePlate3D(hp=0.0003, material=material1, temperature=temp1)

cavity1 = Cavity3D(height=1, r=r, s=s, t=t, medium=fluid1)

lining1 = SinglePlateResonator3D(length=5, depth=1, j=j, l=l, k=k, n=n, plate=plate1, cavity=cavity1)
lining2 = SimpleTwoSidedPlateResonator3D(length=5, depth=1, j=j, l=l, k=k, n=n, plate=plate1, cavity=cavity1)

ductelement1 = DuctElement3D(lining=lining1, medium=fluid1, M=0)
ductelement2 = DuctElement3D(lining=lining2, medium=fluid1, M=0)

duct1 = Duct3D(freq=f, height_d=1, elements=[ductelement1])
duct2 = Duct3D(freq=f, height_d=1, elements=[ductelement2])

# # lining 1 - Numba-Loop test
# start = time.time()
# Z = lining1.zmatrix(1,0,f)
# end = time.time()

# print('Time: ',+ end-start)

# start = time.time()
# Z1 = lining1.zmatrix(1,0,f)
# end = time.time()

# print('Time2: ', + end-start)

# # lining 2 - simple loop
# start = time.time()
# lining2 = SinglePlateResonator3D_2(length=5, depth=1, j=j, l=l, k=k, n=n, plate=plate2, cavity=cavity2) 
# Z3 = lining2.zmatrix(1,0,f)
# end = time.time()

# print('Time3:', + end-start)

# # lining 3 - array method - incomplete

# start = time.time()
# lining3 = SinglePlateResonator3D_3(length=5, depth=1, j=j, l=l, k=k, n=n, plate=plate2, cavity=cavity2)
# Z2 = lining3.zmatrix(1,0,f)
# end = time.time()

# print('Time4:', + end-start)


# %% Test calculate TL and coefficients

# numba loop test
start = time.time()
TL1 = duct1.tl()
TL2 = duct2.tl()
end = time.time()
print('Time4:', + end-start)

start = time.time()
TL1 = duct1.tl()
TL2 = duct2.tl()
end = time.time()
print('Time5:', + end-start)

start = time.time()
[alpha1, beta1, tau1] = duct1.coefficients()
[alpha2, beta2, tau2] = duct2.coefficients()
end = time.time()
print('Time6:', + end-start)

plt.figure()
plt.plot(f,TL1)
plt.plot(f,TL2)
plt.title('Transmission Loss')

labels = ['t', 'r', 'a']

fig, ax = plt.subplots()
ax.stackplot(f, tau1, beta1, alpha1, labels=labels)
ax.legend()
plt.title('Transmissions-, Reflexions-, Absorptionsgrad')


fig, ax = plt.subplots()
ax.stackplot(f, tau2, beta2, alpha2, labels=labels)
ax.legend()
plt.title('Transmissions-, Reflexions-, Absorptionsgrad')
plt.show()

print(alpha1+beta1+tau1)


#%%     

# =============================================================================
# eye1 = np.eye(len(j), len(l))[:, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
# eye2 = np.eye(len(k), len(r), k=1)[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
# eye3 = np.zeros((len(n), len(r)))
# for ni in n:
#     for ri in r:
#         if ni==2*ri:
#             eye3[ni, ri] = 1
#         else:
#             pass
# eye3 = eye3[np.newaxis, np.newaxis, np.newaxis, :, :, np.newaxis, np.newaxis]
# eye = eye1*eye2
# =============================================================================

# =============================================================================
# eye1 = np.eye(len(j), len(l))[:,np.newaxis,:,np.newaxis]
# eye2 = np.eye(len(k), len(n), k=1)[np.newaxis, :, np.newaxis, :]
# eye3 = eye1*eye2
# =============================================================================

