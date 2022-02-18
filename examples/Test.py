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

from acouplasi import DuctElement, DuctElement3D , Duct, Duct3D ,Temperature, Fluid, Material, PlateResonators, SinglePlateResonator,\
SimpleTwoSidedPlateResonator, SinglePlateResonator3D, SimpleTwoSidedPlateResonator3D , Plate, SimplePlate, DoubleLayerPlate,TripleLayerPlate,\
Plate3D, SimplePlate3D, DoubleLayerPlate3D, TripleLayerPlate3D, Cavities2D, Cavity2D, CavityAlt2D, Cavity3D, CavityAlt3D

# from DuctElement import DuctElement, DuctElement3D
# from Ducts import Duct, Duct3D
# from Temperature import Temperature
# from Fluid import Fluid
# from Material import Material
# from Linings import PlateResonators, SinglePlateResonator, SimpleTwoSidedPlateResonator, SinglePlateResonator3D, SimpleTwoSidedPlateResonator3D
# from Plate import Plate, SimplePlate, DoubleLayerPlate, TripleLayerPlate, Plate3D, SimplePlate3D, DoubleLayerPlate3D, TripleLayerPlate3D
# from Cavity import Cavities2D, Cavity2D, CavityAlt2D, Cavity3D, CavityAlt3D

import time
import numba as nb

#%% Test plate resonator 

# =============================================================================
# # frequency
# f = np.arange(10,260,10)

# # temperature
# temp1 = Temperature(C=20)

# # # number of plate modes
# N = 15
 
# # plate modes
# j = np.arange(1, N+1, 1)
# l = np.arange(1, N+1, 1)

# # cavity modes
# r = np.arange(0,30,1)
# t = np.arange(0,30,1)

# fluid1 = Fluid(temperature=temp1)

# material1 = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))
# # Polyurethane TPU1195A
# TPU1195A = Material(rho = 1150, mu = .48, E = lambda freq, temp: 3.1e8*.97**(temp.C-7*np.log10(freq))+1j*4.7e7*.96**(temp.C-7*np.log10(freq)))

# plate1 = SimplePlate(hp=0.0003, material=TPU1195A, temperature=temp1)

# cavity1 = Cavity2D(height=1, r=r, t=t, medium=fluid1)
# #cavity1 = CavityAlt2D(height=1, r=r, medium=fluid1)

# lining1 = SinglePlateResonator(length=5, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)
# #lining1 = SimpleTwoSidedPlateResonator(length=5, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)

# lining3 = SinglePlateResonator(length=10, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)

# ductelement1 = DuctElement(lining=lining1, medium=fluid1, M=0)

# ductelement2 = DuctElement(lining=lining1, medium=fluid1, M=0)

# ductelement3 = DuctElement(lining=lining3, medium=fluid1, M=0)

# duct1 = Duct(freq=f, height_d=1, elements=[ductelement1])

# duct2 = Duct(freq=f, height_d=1, elements=[ductelement1, ductelement2])

# duct3 = Duct(freq=f, height_d=1, elements=[ductelement3])

# Tl = duct1.tl()

# plt.figure()
# plt.plot(f,Tl)
# plt.title('Transmission Loss')

# #%%

# t,r,d = duct1.coefficients()

# labels = ['t', 'r', 'a']

# fig, ax = plt.subplots()
# ax.stackplot(f, t, r, d, labels=labels)
# ax.legend()
# plt.title('Transmissions-, Reflexions-, Absorptionsgrad')
# #plt.show()

# test = d+r+t
# # =============================================================================
# #%% Transfer matrix test

# TL2 =  duct1.tl2()

# plt.figure()
# plt.plot(f, TL2, f, Tl, '-.')
# plt.title('Transmission Loss')
# plt.xlabel('f in Hz')
# plt.ylabel('TL in dB')
# plt.legend(['with transfer matrix', 'without transfer matrix'])
# plt.show()


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
#cavity1 = CavityAlt3D(height=1, r=r, s=s, medium=fluid1)

lining1 = SinglePlateResonator3D(length=5, depth=1, j=j, l=l, k=k, n=n, t=t, plate=plate1, cavity=cavity1)
#lining1 = SimpleTwoSidedPlateResonator3D(length=5, depth=1, j=j, l=l, k=k, n=n, t=t, plate=plate1, cavity=cavity1)

ductelement1 = DuctElement3D(lining=lining1, medium=fluid1, M=0)

duct1 = Duct3D(freq=f, height_d=1, elements=[ductelement1])

# Calculation

start = time.time()
TL1 = duct1.tl()
end = time.time()
print('Time 1:', + end-start)

start = time.time()
TL1 = duct1.tl()
end = time.time()
print('Time 2:', + end-start)

start = time.time()
[tra, ref, dis] = duct1.coefficients()
end = time.time()
print('Time3:', + end-start)

plt.figure()
plt.plot(f,TL1)
plt.title('Transmission Loss')

labels = ['t', 'r', 'a']

fig, ax = plt.subplots()
ax.stackplot(f, tra, ref, dis, labels=labels)
ax.legend()
plt.title('Transmissions-, Reflexions-, Absorptionsgrad')


#%% Transfermatrix test

TL2 = duct1.tl2()

plt.figure()
plt.plot(f, TL2, f, TL1, '-.')
plt.title('Transmission Loss')
plt.xlabel('f in Hz')
plt.ylabel('TL in dB')
plt.legend(['with transfer matrix', 'without transfer matrix'])
plt.show()

#%% Vergleich mit 2D

# # temperature
# temp2 = Temperature(C=20)

# # # number of plate modes
# N = 15
 
# # plate modes
# j = np.arange(1, N+1, 1)
# l = np.arange(1, N+1, 1)

# # cavity modes
# r = np.arange(0,30,1)
# t = np.arange(0,30,1)

# fluid2 = Fluid(temperature=temp2)

# material2 = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))
# # Polyurethane TPU1195A
# TPU1195A = Material(rho = 1150, mu = .48, E = lambda freq, temp: 3.1e8*.97**(temp.C-7*np.log10(freq))+1j*4.7e7*.96**(temp.C-7*np.log10(freq)))

# plate2 = SimplePlate(hp=0.0003, material=material2, temperature=temp2)

# cavity2 = Cavity2D(height=1, r=r, t=t, medium=fluid2)
# #cavity1 = CavityAlt2D(height=1, r=r, medium=fluid1)

# lining2 = SinglePlateResonator(length=5, depth=1, j=j, l=l, t=t, plate=plate2, cavity=cavity2)

# ductelement2 = DuctElement(lining=lining2, medium=fluid2, M=0)

# duct2 = Duct(freq=f, height_d=1, elements=[ductelement2])

# TL2 = duct2.tl()

# fig = plt.figure()
# plt.plot(f, TL1)
# plt.plot(f, TL2, '--')
# plt.show()
