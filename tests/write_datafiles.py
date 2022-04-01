# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) 2021,
#------------------------------------------------------------------------------
"""Implements writing of Testdatafile for the silencers.
   Only run this after major fixes in the Code.
"""

import numpy as np

from acouplasi import  DuctElement3D , Duct3D ,Temperature, Fluid, Material,\
SinglePlateResonator3D, SimplePlate3D,  Cavity3D, DuctElement , Duct ,  SinglePlateResonator,\
SimpleTwoSidedPlateResonator , SimplePlate, Cavity2D


# frequency
f = np.arange(10, 260, 10)
# temperature
temp1 = Temperature(C=20)
# # number of plate modes
N = 1
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


np.save('silencer3d.npy', [duct1.tl(),duct1.coefficients()]) 

#%%
####### 2D here

# # frequency
f = np.arange(10,260,10)
# # temperature
temp1 = Temperature(C=20)
#number of plate modes
N = 5
# plate modes
j = np.arange(1, N+1, 1)
l = np.arange(1, N+1, 1)
# cavity modes
r = np.arange(0,30,1)
t = np.arange(0,30,1)
fluid1 = Fluid(temperature=temp1)

material1 = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))
#Polyurethane TPU1195A
TPU1195A = Material(rho = 1150, mu = .48, E = lambda freq, temp: 3.1e8*.97**(temp.C-7*np.log10(freq))+1j*4.7e7*.96**(temp.C-7*np.log10(freq)))

plate1 = SimplePlate(hp=0.0003, material=TPU1195A, temperature=temp1)

cavity1 = Cavity2D(height=1, r=r, t=t, medium=fluid1)

lining1 = SinglePlateResonator(length=5, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)
lining2 = SimpleTwoSidedPlateResonator(length=5, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)
lining3 = SinglePlateResonator(length=10, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)

ductelement1 = DuctElement(lining=lining1, medium=fluid1, M=0)
ductelement2 = DuctElement(lining=lining2, medium=fluid1, M=0)
ductelement3 = DuctElement(lining=lining3, medium=fluid1, M=0)

duct1 = Duct(freq=f, height_d=1, elements=[ductelement1])
duct2 = Duct(freq=f, height_d=1, elements=[ductelement1, ductelement2])
duct3 = Duct(freq=f, height_d=1, elements=[ductelement3])

np.save('silencer2d.npy', [duct1.tl(),duct1.coefficients(),duct2.tl(),duct2.coefficients(),duct3.tl(),duct3.coefficients()]) 



