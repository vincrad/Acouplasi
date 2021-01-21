#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 11:08:32 2021

@author: radmann
"""

#%% Mockup-Test

import numpy as np, numpy
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

cavity1 = Cavity(height=0.5, R=R, S=S, medium=fluid1)

lining1 = SinglePlateResonator(length=1, J=J, L=L, plate=plate1, cavity=cavity1)

ductelement1 = DuctElement(lining=lining1, medium=fluid1)

duct1 = Duct(elements=[ductelement1], freq=f)

transmissionloss = duct1.tl()





















#%% Test-Vektorisierung

freq = np.arange(10,260,10)

omega = 2*np.pi*freq
        
k0 = omega/fluid1.c

M = 0

L_new = L[:, np.newaxis, np.newaxis, np.newaxis]
J_new = J[np.newaxis, :, np.newaxis, np.newaxis]


R_new = R[np.newaxis, np.newaxis, :, np.newaxis]



# incident sound
# =============================================================================
# I = np.zeros((len(self.lining.L), len(omega)), dtype=complex)
# 
# 
# for l in self.lining.L:
# 
#     x0=self.lining.length**2*k0**2
#     x1=numpy.pi**2*l**2
#     x2=numpy.pi*self.lining.length*l
#     x3=numpy.exp(1j*self.lining.length*k0)
# 
#     I[l-1,:] = (-1)**l*x2/(x0*x3 - x1*x3) - x2/(x0 - x1)
#     
# return I
# 
# =============================================================================

# new code
x0=lining1.length**2*k0**2
x1=numpy.pi**2*L_new**2
x2=numpy.pi*lining1.length*L_new
x3=numpy.exp(1j*lining1.length*k0)

I_new = (-1)**L_new*x2/(x0*x3 - x1*x3) - x2/(x0 - x1)



# Zprad
# =============================================================================
# for j in self.J:
#             
#             for l in self.L:
#                 
#                 Sum = 0
#                 
#                 if j==l:
#                     
#                     Sum += 0
#                     
#                 else:
#                     
#                     for r in self.cavity.R:
#                         
#                         Krp = -k0*self.M-1j*np.sqrt((1-self.M**2)*(r*np.pi)**2-k0**2, dtype=complex)/(1-self.M**2)
#                         Krm = k0*self.M-1j*np.sqrt((1-self.M**2)*(r*np.pi)**2-k0**2, dtype=complex)/(1-self.M**2)
#                         
#                         x0=numpy.pi**2
#                         x1=self.length**2
#                         x2=Krm**2*x1
#                         x3=j**2
#                         x4=-x0*x3
#                         x5=l**2
#                         x6=-x0*x5
#                         x7=x2 + x6
#                         x8=j + 1
#                         x9=(-1)**(j + l)
#                         x10=1j*self.length
#                         x11=Krm*x10
#                         x12=numpy.exp(x11)
#                         x13=x3 - x5
#                         x14=x0*x13
#                         x15=j*k0*l*x1/x13
#                         x16=Krp**2*x1
#                         x17=x16 + x6
#                         x18=Krp*x10
#                         x19=numpy.exp(x18)
#                         
#                         Sum += (2 - self.cavity.delta(r))*(x15*(x12*x7*((-1)**(l + x8) + 1) + x14*((-1)**x8 + x12*x9))*numpy.exp(-x11)/(x7*(x2 + x4)) + x15*((-1)**(l + 1)*x14 + x14*x19 + x17*x19*(x9 - 1))*numpy.exp(-x18)/(x17*(x16 + x4)))/numpy.sqrt(-k0**2 + r**2*x0, dtype=complex)
#                         
#                 Zprad[j-1,l-1,:] = (1/2)*1j*self.length*Sum
# =============================================================================


Krp = -k0*M-1j*np.sqrt((1-M**2)*(R_new*np.pi)**2-k0**2, dtype=complex)/(1-M**2)
Krm = k0*M-1j*np.sqrt((1-M**2)*(R_new*np.pi)**2-k0**2, dtype=complex)/(1-M**2)

x0=numpy.pi**2
x1=lining1.length**2
x2=Krm**2*x1
x3=J_new**2
x4=-x0*x3
x5=L_new**2
x6=-x0*x5
x7=x2 + x6
x8=J_new + 1
x9=(-1)**(J_new + L_new)
x10=1j*lining1.length
x11=Krm*x10
x12=numpy.exp(x11)
x13=x3 - x5
x14=x0*x13
#x15=J_new*k0*L_new*x1/x13
x15 = np.divide(J_new*k0*L_new*x1, x13, out=np.zeros_like(J_new*k0*L_new*x1), where=x13!=0)
x16=Krp**2*x1
x17=x16 + x6
x18=Krp*x10
x19=numpy.exp(x18)

deltar = np.eye(5,1)[np.newaxis, np.newaxis, :]

Sum = (2 - deltar)*(x15*(x12*x7*((-1)**(L_new + x8) + 1) + x14*((-1)**x8 + x12*x9))*numpy.exp(-x11)/(x7*(x2 + x4)) + x15*((-1)**(L_new + 1)*x14 + x14*x19 + x17*x19*(x9 - 1))*numpy.exp(-x18)/(x17*(x16 + x4)))/numpy.sqrt(-k0**2 + R_new**2*x0, dtype=complex)

Erg = np.sum(Sum, axis=2)*(1/2)*1j*lining1.length


# =============================================================================
# factor = 0.9
# eps=1
# while (1+eps>1):
#     eps = eps*factor
# print(eps)
# 
# =============================================================================


# Zc
# =============================================================================
# for j in J:
#             
#             for l in L:
#                 
#                 Sum = 0
#                 
#                 for r in self.R:
#                     
#                     if j==r or l==r:
#                         
#                         Sum += 0
#                         
#                     else:
#                         
#                         for s in self.S:
#                             
#                             x0=numpy.pi*r**2
#                             x1=self.length*j/(-numpy.pi*j**2 + x0)
#                             x2=(-1)**r
#                             x3=self.length*l/(-numpy.pi*l**2 + x0)
#                      
#                             Sum += 1j*omega*(2 - self.delta(r))*(2 - self.delta(s))*((-1)**j*x1*x2 - x1)*((-1)**l*x2*x3 - x3)/(self.height*(-k0**2 + 2*1j*k0*self.kappa(r, s)*self.zetars + self.kappa(r, s)**2))
#                     
#                 Zc[j-1,l-1,:] = Sum
# =============================================================================

L_new2 = L[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
J_new2 = J[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
R_new2 = R[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
S_new2 = S[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]


# =============================================================================
# kappa = np.sqrt((r*np.pi/self.length)**2+(s*np.pi/self.height)**2)
# =============================================================================

kappa = np.sqrt((R_new2*np.pi/lining1.length)**2+(S_new2*np.pi/cavity1.height)**2)


x0=numpy.pi*R_new2**2
#x1=lining1.length*J_new2/(-numpy.pi*J_new2**2 + x0)
#x1=np.nan_to_num(lining1.length*J_new2/(-numpy.pi*J_new2**2 + x0), posinf=0.0, neginf=0.0)
x1 = np.divide(lining1.length*J_new2, -numpy.pi*J_new2**2 + x0, out=np.zeros_like(J_new2*R_new2, dtype=float), where=(-numpy.pi*J_new2**2 + x0)!=0)
x2=(-1)**R_new2
#x3=lining1.length*L_new2/(-numpy.pi*L_new2**2 + x0)
x3 = np.divide(lining1.length*L_new2, -numpy.pi*L_new2**2 + x0, out=np.zeros_like(L_new2*R_new2, dtype=float), where=(-numpy.pi*L_new2**2 + x0)!=0)

deltar = np.eye(5,1)[np.newaxis, np.newaxis, :, np.newaxis]
deltas = np.eye(5,1)[np.newaxis, np.newaxis, np.newaxis, :]

Sum = 1j*omega*(2 - deltar)*(2 - deltas)*((-1)**J_new2*x1*x2 - x1)*((-1)**L_new2*x2*x3 - x3)/(cavity1.height*(-k0**2 + 2*1j*k0*kappa*cavity1.zetars + kappa**2))

Zcav = np.sum(Sum, axis=(2,3))




# =============================================================================
# np.ravel_multi_inde
# np.unique
# =============================================================================






