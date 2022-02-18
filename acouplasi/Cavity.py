#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:49:08 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
import numba as nb

from .Fluid import Fluid


class Cavities2D(tr.HasTraits):
    
    '''
    Parental class for different types of cavities. 
    '''
    
    # height of the cavity
    height = tr.Float()
    
    # modes of the cavity
    # x direction
    r = tr.Instance(np.ndarray)
    
    # medium
    medium = tr.Instance(Fluid)

class Cavity2D(Cavities2D):
    
    '''
    Class to define a 2D cavity for a plate silencer according to Wang.
    '''   
    
    # modes of the cavity
    # z-axis
    t = tr.Instance(np.ndarray)
    
    # loss factor
    zetart = tr.Float(default_value=0.1)
    
    # property methods to define expanded arrays of modes
    @property
    def R(self):
        
        return self.r[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        
    
    @property
    def T(self):
        
        return self.t[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
       
    
    # property methods to define the Kronecker delta
    @property
    def deltar(self):    
        
        return np.eye(len(self.r),1)[np.newaxis, np.newaxis, :, np.newaxis]
    
    
    @property
    def deltat(self):
        
        return np.eye(len(self.t),1)[np.newaxis, np.newaxis, np.newaxis, :]
        
    
    # method to calculate kappars
    def kappart(self, length):
        
        return np.sqrt((self.R*np.pi/length)**2+(self.T*np.pi/self.height)**2)
        
        
    
    # method calculate the cavity impedance matrix
    def cavityimpedance(self, length, depth, j, l, freq):
        
        # define expanded arrays of modes
        L = l[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        J = j[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c
        
        # calculate the cavity impedance matrix
        x0=numpy.pi*self.R**2
        #x1=length*J/(-numpy.pi*J**2 + x0)
        x1 = np.divide(length*J, (-numpy.pi*J**2 + x0), out=np.zeros_like(J*self.R, dtype=float), where=(-numpy.pi*J**2 + x0)!=0)
        x2=(-1)**self.R
        #x3=length*L/(-numpy.pi*L**2 + x0)
        x3 = np.divide(length*L, (-numpy.pi*L**2 + x0), out=np.zeros_like(L*self.R, dtype=float), where=(-numpy.pi*L**2 + x0)!=0)
        
        Zc_temp = 1j*omega*self.medium.rho0*(2 - self.deltar)*(2 - self.deltat)*((-1)**J*x1*x2 - x1)*((-1)**L*x2*x3 - x3)/(length*self.height*(-k0**2 + 2*1j*k0*self.kappart(length)*self.zetart + self.kappart(length)**2))
        
        # building the final Zc matrix by summation over R and T
        Zc = np.sum(Zc_temp, axis=(2,3))
        
        return Zc
    
class CavityAlt2D(Cavities2D):
    
    '''
    Class to define a 2D cavity, derived from a rectangular duct, for a plate silencer.
    '''
    
    # property methods to define expanded arrays of modes
    @property
    def R(self):
        
        return self.r[np.newaxis, np.newaxis, :, np.newaxis]
    
    # property methods to define the Kronecker delta
    @property
    def deltar(self):    
        
        return np.eye(len(self.r),1)[np.newaxis, np.newaxis, :]
    
    # method calculate the cavity impedance matrix
    def cavityimpedance(self, length, depth, j, l, freq):
        
        # define expanded arrays of modes
        L = l[:, np.newaxis, np.newaxis, np.newaxis]
        J = j[np.newaxis, :, np.newaxis, np.newaxis]
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c
        
        # calculate the cavity impedance matrix
        x0 = np.pi**2
        x1 = self.R**2
        x2 = -x1
        
        kr = np.sqrt(k0**2 - x0*x1/length**2)
        
        Zc_temp = 1j*length*omega*self.medium.rho0*(self.deltar - 2)/(kr*x0*np.tan(self.height*kr))*np.divide(J*((-1)**(J + self.R) - 1),(J**2 + x2), out=np.zeros_like(J*self.R, dtype=float), where=J!=self.R)*np.divide(L*((-1)**(L + self.R) - 1),(L**2 + x2), out=np.zeros_like(L*self.R, dtype=float), where=L!=self.R)

        # building the final Zc matrix by summation over R
        Zc = np.sum(Zc_temp, axis=2)
        
        return Zc
#%% 3D Cavity

class Cavities3D(tr.HasTraits):

    # height of the cavity
    height = tr.Float()
    
    # modes of the cavity
    # x direction
    r = tr.Instance(np.ndarray)
    # y direction
    s = tr.Instance(np.ndarray)
    
    # medium
    medium = tr.Instance(Fluid)


## Cavity3D loop
# method to calculate the cavity impedance matrix 
# accelerated by numba
@nb.njit
def get_zc(c, rho0, zetarst, j, k, l, n, r, s, t, length, depth, height, freq):
    
    # circular frequency and wave number
    omega = 2*np.pi*freq
    k0 = omega/c
    
    # define empty cavity impedance matrix
    Zc = np.zeros((len(j), len(k), len(l), len(n), len(freq)), dtype=np.complex128)
    
    for J in j:
        
        for K in k:
            
            for L in l:
                
                for N in n:
                    
                    # define empty sum array
                    Sum = np.zeros(len(freq), dtype=np.complex128)
                    
                    for R in r:
                        
                        for S in s:
                            
                            for T in t:
                                
                                # calculate the Kronecker Deltas
                                deltar = 1 if R==0 else 0
                                deltas = 1 if S==0 else 0
                                deltat = 1 if T==0 else 0
                                
                                # calculate kapparst
                                kapparst = np.sqrt((R*np.pi/length)**2+(S*np.pi/depth)**2+ (T*np.pi/height)**2)
                                
                                # j=r
                                if J==R:
                                    
                                    Sum += 0
                                    
                                # k=s
                                elif K==S:
                                    
                                    Sum += 0
                                
                                # l=r
                                elif L==R:
                                    
                                    Sum += 0
                                
                                # n=s
                                elif N==S:
                                    
                                    Sum += 0
                                    
                                
                                # else
                                else:
                                    
                                    x0=numpy.pi*(R)**(2+0j)
                                    x1=numpy.pi*(S)**(2+0j)
                                    x2=length*depth
                                    x3=J*K*x2/((-numpy.pi*(J)**(2+0j) + x0)*(-numpy.pi*(K)**(2+0j) + x1))
                                    x4=(-1)**(R+0j)
                                    x5=(-1)**(J+0j)*x3*x4
                                    x6=(-1)**(S+0j)
                                    x7=(-1)**(K+0j)*x6
                                    x8=L*N*x2/((-numpy.pi*(L)**(2+0j) + x0)*(-numpy.pi*(N)**(2+0j) + x1))
                                    x9=(-1)**(L+0j)*x4*x8
                                    x10=(-1)**(N+0j)*x6
                                    
                                    Sum += 1j*omega*rho0*(2 - deltar)*(2 - deltas)*(2 - deltat)*(-x10*x8 + x10*x9 + x8 - x9)*(-x3*x7 + x3 + x5*x7 - x5)/(length*depth*height*(-(k0)**(2+0j) + 2*1j*k0*kapparst*zetarst + (kapparst)**(2+0j)))
                                    
                    Zc[J-1, K-1, L-1, N-1, :] = Sum
    
    return Zc

class Cavity3D(Cavities3D):
    
    '''
    Class to define a 3D cavity for a plate silencer.
    '''
    
    # modes of the cavity
    # z direction
    t = tr.Instance(np.ndarray)
    
    # loss factor
    zetarst = tr.Float(default_value=0.1)
    
    # method calls the method which calculate the cavity impedance matrix
    # accelerated by numba
    def cavityimpedance(self, length, depth, j, k, l, n, freq):
        
        Zc = get_zc(self.medium.c, self.medium.rho0, self.zetarst, j, k, l, n, self.r, self.s, self.t, length, depth, self.height, freq)
        
        return Zc


class CavityAlt3D(Cavities3D):
    
    '''
    Class to define a 3D cavity for a plate silencer.
    '''
    
    # property methods to define expanded arrays of modes
    @property
    def R(self):
        
        return self.r[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    
    @property
    def S(self):
        
        return self.s[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    
    # property methods to define the Kronecker deltas
    @property
    def deltar(self):    
        
        return np.eye(len(self.r),1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    
    @property
    def deltas(self):    
        
        return np.eye(len(self.s),1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]

    # method calculate the cavity impedance matrix
    def cavityimpedance(self, length, depth, j, k, l, n, freq):
        
        # define expanded arrays of modes
        J = j[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        K = k[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        L = l[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        N = n[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c
        
        # calculate the cavity impedance matrix
        x0 = np.pi**2
        x1 = self.S**2
        x2 = self.R**2
        x3 = -x2
        x4 = -x1
        x5 = J + self.R
        x6 = K + self.S
        x7 = L + self.R
        x8 = N + self.S
        
        krs = np.sqrt(k0**2 - x0*x2/length**2 - x0*x1/depth**2)
        
        Zc_temp = -1j*length*depth*J*K*L*N*omega*self.medium.rho0*(self.deltar - 2)*(self.deltas - 2)*np.divide((-(-1)**x5 - (-1)**x6 + (-1)**(x5 + x6) + 1)*(-(-1)**x7 - (-1)**x8 + (-1)**(x7 + x8) + 1),(J**2 + x3)*(K**2 + x4)*(L**2 + x3)*(N**2 + x4), out=np.zeros_like(J*L*K*N*self.R*self.S, dtype=float), where= np.logical_and(np.logical_and(J!=self.R, L!=self.R), np.logical_and(K!=self.S, N!=self.S)))/(np.pi**4*krs*np.tan(self.height*krs))
        
        # building the final cavity impedance matrix by summation over R, S
        Zc = np.sum(Zc_temp, axis=(4,5))
        
        return Zc            
