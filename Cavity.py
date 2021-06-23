#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:49:08 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
from Fluid import Fluid

#%% 2D Cavities

class Cavities2D(tr.HasTraits):
    
    '''
    Parental class for different types of cavities. 
    '''
    
    # height of the cavity
    height = tr.Float()
    
    # medium
    medium = tr.Instance(Fluid)

class Cavity2D(Cavities2D):
    
    '''
    Class to define a 2D cavity for a plate silencer according to Wang.
    '''   
    
    # modes of the cavity
    r = tr.Instance(np.ndarray)
    s = tr.Instance(np.ndarray)
    
    # loss factor
    zetars = tr.Float(default_value=0.1)
    
    # property methods to define expanded arrays of modes
    @property
    def R(self):
        
        return self.r[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
        
    
    @property
    def S(self):
        
        return self.s[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
       
    
    # property methods to define the Kronecker delta
    @property
    def deltar(self):    
        
        return np.eye(len(self.r),1)[np.newaxis, np.newaxis, :, np.newaxis]
    
    
    @property
    def deltas(self):
        
        return np.eye(len(self.s),1)[np.newaxis, np.newaxis, np.newaxis, :]
        
    
    # method to calculate kappars
    def kappars(self, length):
        
        return np.sqrt((self.R*np.pi/length)**2+(self.S*np.pi/self.height)**2)
        
        
    
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
        
        Zc_temp = 1j*omega*self.medium.rho0*(2 - self.deltar)*(2 - self.deltas)*((-1)**J*x1*x2 - x1)*((-1)**L*x2*x3 - x3)/(length*self.height*(-k0**2 + 2*1j*k0*self.kappars(length)*self.zetars + self.kappars(length)**2))
        
        # building the final Zc matrix by summation over R and S
        Zc = np.sum(Zc_temp, axis=(2,3))
        
        return Zc
    
class CavityAlt2D(Cavities2D):
    
    '''
    Class to define a 2D cavity, derived from a rectangular duct, for a plate silencer.
    '''
    
    # modes of the cavity
    r = tr.Instance(np.ndarray)
    
    # property methods to define expanded arrays of modes
    @property
    def R(self):
        
        return self.r[np.newaxis, np.newaxis, :, np.newaxis]
    
    # property methods to define the Kronecker delta
    @property
    def deltar(self):    
        
        deltar=np.zeros(len(self.r))
        deltar[0]=1

        return deltar[np.newaxis, np.newaxis, :, np.newaxis]
    
    # method to calculate wave number kr
    def kr(self, k0, length):
        
        return np.sqrt(k0**2 - (np.pi*self.R/length)**2)
    
    # method calculate the cavity impedance matrix
    def cavityimpedance(self, length, depth, j, l, freq):
        
        # define expanded arrays of modes
        L = l[:, np.newaxis, np.newaxis, np.newaxis]
        J = j[np.newaxis, :, np.newaxis, np.newaxis]
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c
        
        # calculate the cavity impedance matrix
        Zc_temp = 1j*length*omega*self.medium.rho0*(self.deltar - 2)/(np.pi**2*self.kr(k0, length)*np.tan(self.height*self.kr(k0, length)))*np.divide(L*((-1)**(L + self.R) - 1), (L**2 - self.R**2), out=np.zeros_like(L*self.R, dtype=float), where=L!=self.R)*np.divide(J*((-1)**(J + self.R) - 1), (J**2 - self.R**2), out=np.zeros_like(J*self.R, dtype=float), where=J!=self.R)
        
        # building the final Zc matrix by summation over R
        Zc = np.sum(Zc_temp, axis=2)
        
        return Zc
#%% 3D Cavity

class Cavity3D(tr.HasTraits):
    
    '''
    Class to define a 3D cavity for a plate silencer.
    '''
    
    # height of the cavity
    height = tr.Float()
    
    # modes of the cavity
    # x direction
    r = tr.Instance(np.ndarray)
    # y direction
    s = tr.Instance(np.ndarray)
    # z direction
    t = tr.Instance(np.ndarray)
    
    # loss factor
    zetarst = tr.Float(default_value=0.1)
    
    # medium
    medium = tr.Instance(Fluid)
    
    # property methods to define expanded arrays of modes
    @property
    def R(self):
        
        return self.r[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    
    @property
    def S(self):
        
        return self.s[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    
    @property
    def T(self):
        
        return self.t[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    
    # property methods to define the Kronecker deltas
    @property
    def deltar(self):    
        
        return np.eye(len(self.r),1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
    
    @property
    def deltas(self):    
        
        return np.eye(len(self.s),1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    
    @property
    def deltat(self):    
        
        return np.eye(len(self.t),1)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    
    # method to calculate the kappa_rst
    def kapparst(self, length, depth):
        
        return np.sqrt((self.R*np.pi/length)**2+(self.S*np.pi/depth)**2+ (self.T*np.pi/self.height)**2)

    # method calculate the cavity impedance matrix
    def cavityimpedance(self, length, depth, j, k, l, n, freq):
        
        # define expanded arrays of modes
        J = j[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        K = k[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        L = l[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        N = n[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c
        
        # calculate the cavity impedance matrix
# =============================================================================
#         x0=-self.R**2
#         x1=-self.S**2
#         x2=J + self.R
#         x3=K + self.S
#         x4=L + self.R
#         x5=N + self.S
#         
#         Zc_temp = 1j*length*depth*J*K*L*N*omega*self.medium.rho0*(2 - self.deltar)*(2 - self.deltas)*(2 - self.deltat)*(-(-1)**x2 - (-1)**x3 + (-1)**(x2 + x3) + 1)*(-(-1)**x4 - (-1)**x5 + (-1)**(x4 + x5) + 1)/(numpy.pi**4*self.height*(J**2 + x0)*(K**2 + x1)*(L**2 + x0)*(N**2 + x1)*(-k0**2 + 2*1j*k0*self.kapparst(length, depth)*self.zetarst + self.kapparst(length, depth)**2))
# =============================================================================
        
# =============================================================================
#         x0=-self.R**2
#         x1=-self.S**2
#         x2=J + self.R
#         x3=K + self.S
#         x4=L + self.R
#         x5=N + self.S
#         x6 = np.divide(1, J**2 + x0, out=np.zeros_like(J*self.R, dtype=float), where=J**2 + x0!=0)
#         x7 = np.divide(1, K**2 + x1, out=np.zeros_like(K*self.S, dtype=float), where=K**2 + x1!=0)
#         x8 = np.divide(1, L**2 + x0, out=np.zeros_like(L*self.R, dtype=float), where=L**2 + x0!=0)
#         x9 = np.divide(1, N**2 + x1, out=np.zeros_like(N*self.S, dtype=float), where=N**2 + x1!=0)
#         
#         Zc_temp = 1j*length*depth*J*K*L*N*omega*self.medium.rho0*(2 - self.deltar)*(2 - self.deltas)*(2 - self.deltat)*(-(-1)**x2 - (-1)**x3 + (-1)**(x2 + x3) + 1)*(-(-1)**x4 - (-1)**x5 + (-1)**(x4 + x5) + 1)/(numpy.pi**4*self.height*(-k0**2 + 2*1j*k0*self.kapparst(length, depth)*self.zetarst + self.kapparst(length, depth)**2))*x6*x7*x8*x9
# =============================================================================


        x0=numpy.pi*self.R**2
        x1=numpy.pi*self.S**2
        x2=length*depth
        #x3=J*K*x2/((-numpy.pi*J**2 + x0)*(-numpy.pi*K**2 + x1))
        x3 = np.divide(J*K*x2, ((-numpy.pi*J**2 + x0)*(-numpy.pi*K**2 + x1)), out=np.zeros_like(J*self.R*K*self.S, dtype=float), where=((-numpy.pi*J**2 + x0)*(-numpy.pi*K**2 + x1))!=0)
        x4=(-1)**self.R
        x5=(-1)**J*x3*x4
        x6=(-1)**self.S
        x7=(-1)**K*x6
        #x8=L*N*x2/((-numpy.pi*L**2 + x0)*(-numpy.pi*N**2 + x1))
        x8 = np.divide(L*N*x2, ((-numpy.pi*L**2 + x0)*(-numpy.pi*N**2 + x1)), out=np.zeros_like(L*self.R*N*self.S, dtype=float), where=((-numpy.pi*L**2 + x0)*(-numpy.pi*N**2 + x1))!=0)
        x9=(-1)**L*x4*x8
        x10=(-1)**N*x6
        
        Zc_temp = 1j*omega*self.medium.rho0*(2 - self.deltar)*(2 - self.deltas)*(2 - self.deltat)*(-x10*x8 + x10*x9 + x8 - x9)*(-x3*x7 + x3 + x5*x7 - x5)/(length*depth*self.height*(-k0**2 + 2*1j*k0*self.kapparst(length, depth)*self.zetarst + self.kapparst(length, depth)**2))
        
        # building the final cavity impedance matrix by summation over R, S, T
        Zc = np.sum(Zc_temp, axis=(4,5,6))
        
        return Zc
                         
                            
        
            
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    