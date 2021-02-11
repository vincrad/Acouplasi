#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:58:19 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
from Fluid import Fluid
from Cavity import Cavity
from Plate import Plate


class Linings(tr.HasTraits):
    
    '''
    Parental class for different kinds of linings.
    '''
    
    # geometry
    length = tr.Float()
    depth = tr.Float(default_value=1)
    
    
class PlateResonators(Linings):
    
    '''
    Parental class for different types of plate resonator configurations.
    '''
    
    # plate modes
    j = tr.Instance(np.ndarray)
    l = tr.Instance(np.ndarray)
    

class SinglePlateResonator(PlateResonators):
    
    '''
    2D single plate resonatr without flow.
    '''
    
    # plate
    plate = tr.Instance(Plate)
    
    # cavity
    cavity = tr.Instance(Cavity)
    
    # property method to define the Kronecker delta
    @property
    def deltar(self):
        
        return np.eye(len(self.cavity.r),1)[np.newaxis, np.newaxis, :]
        
    
    # methode calculate the impedance matrix of the plate resonator
    def zmatrix(self, M, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.cavity.medium.c
        
        # cavity impedance matrix
        Zc = self.cavity.cavityimpedance(self.length, self.depth, self.j, self.l, freq)
        
        # impedance matrix of the plate induced sound
        # define expanded arrays of modes
        L = self.l[:, np.newaxis, np.newaxis, np.newaxis]
        J = self.j[np.newaxis, :, np.newaxis, np.newaxis]
        R = self.cavity.r[np.newaxis, np.newaxis, :, np.newaxis]
        
        # calculate the impedance matrix of the plate induced sound
        Krp = -k0*M-1j*np.sqrt((1-M**2)*(R*np.pi)**2-k0**2, dtype=complex)/(1-M**2)
        Krm = k0*M-1j*np.sqrt((1-M**2)*(R*np.pi)**2-k0**2, dtype=complex)/(1-M**2)
        
        # for j=l
        x0=numpy.pi**2
        x1=self.length**2
        x2=L**2*x0
        x3=-x2
        x4=Krm**2*x1 + x3
        x5=Krm*M
        x6=1j*self.length
        x7=Krm*x6
        x8=numpy.exp(x7)
        x9=(-1)**(L + 1) + x8
        x10=2*self.length*x2
        x11=omega*x10
        x12=1j*x4*x8
        x13=omega*x1
        x14=M*x2
        x15=(1/2)*self.length/omega
        x16=Krp**2*x1 + x3
        x17=Krp*M
        x18=(-1)**L
        x19=x10*x17
        x20=Krp*x6
        x21=numpy.exp(x20)
        x22=1j*x16*x21
        
        Zpll_temp = (1/2)*x6*(2 - self.deltar)*(x15*(k0 + x5)*(-Krm*x12*x13 + x10*x5*x9 + x11*x9 - x12*x14)*numpy.exp(-x7)/x4**2 + x15*(k0 - x17)*(-Krp*x13*x22 - x11*x18 + x11*x21 + x14*x22 + x18*x19 - x19*x21)*numpy.exp(-x20)/x16**2)/numpy.sqrt(-k0**2 + R**2*x0*(1 - M**2), dtype=complex)
        
        Zpll = np.sum(Zpll_temp*(np.identity(len(self.l))[:, :, np.newaxis, np.newaxis]), axis=2)
        
        # for j != l
        x0=numpy.pi**2
        x1=self.length**2
        x2=Krm**2*x1
        x3=J**2
        x4=-x0*x3
        x5=L**2
        x6=-x0*x5
        x7=x2 + x6
        x8=Krm*M
        x9=J + 1
        x10=(-1)**(J + L)
        x11=1j*self.length
        x12=Krm*x11
        x13=numpy.exp(x12)
        x14=(-1)**x9 + x10*x13
        x15=x3 - x5
        x16=x0*x15
        x17=omega*x16
        x18=(-1)**(L + x9) + 1
        x19=x13*x18*x7
        #x20=J*L*x1/(omega*x15)
        x20 = np.divide(J*L*x1, (omega*x15), out=np.zeros_like(J*L*x1/omega), where=(omega*x15)!=0)
        x21=Krp**2*x1
        x22=x21 + x6
        x23=Krp*M
        x24=x16*x23
        x25=Krp*x11
        x26=numpy.exp(x25)
        x27=x22*x26
        
        Zplj_temp = (1/2)*x11*(2 - self.deltar)*(x20*(k0 + x8)*(omega*x19 + x14*x16*x8 + x14*x17 + x19*x8)*numpy.exp(-x12)/(x7*(x2 + x4)) + x20*(k0 - x23)*((-1)**L*x24 + (-1)**(L + 1)*x17 + omega*x27*(x10 - 1) + x17*x26 + x18*x23*x27 - x24*x26)*numpy.exp(-x25)/(x22*(x21 + x4)))/numpy.sqrt(-k0**2 + R**2*x0*(1 - M**2), dtype=complex)
        
        Zplj = np.sum(Zplj_temp, axis=2)
        
        # calculate the overall impedance matrix of the plate induced sound
        Zprad = Zplj+Zpll
        
        # calculate the total impedance matrix of the lining
        Z = Zc+Zprad
        
        return Z
   
    # method calculate the plate velocity
    def platevelocity(self, I, M, freq):
        
        # impedance matrix
        Z = self.zmatrix(M, freq)
        
        # L matrix with material properties of the plate
        Lmatrix = self.plate.lmatrix(self.length, self.depth, self.l, freq)
        
        # solving the linear system of equations
        # building lhs matrix from Z and Lmatrix
        lhs = Z+Lmatrix
        
        # calculate the plate velocity
        vp = np.zeros_like(I, dtype=complex)
        for i in range(len(freq)):
            
            vp[:,i] = np.linalg.solve(lhs[:,:,i], I[:,i])
        
        return vp
        
    # method calculates the transmission loss of the plate silencer
    def transmissionloss(self, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis]
        
        # plate velocity
        vp = self.platevelocity(I, M, freq)
        
        # calculate transmission loss
        x0=self.length**2
        x1=numpy.pi**2*J**2
        x2=M*omega
        x3=M*k0
        x4=numpy.exp(1j*self.length*k0/(M + 1))
        x5=(-1)**(J + 1)*x4
        
        temp = (1/2)*numpy.pi*J*x0*((-1)**J*x3*x4 + omega*x5 + omega + x2*x5 + x2 - x3)/(omega*(M**2*x1 + 2*M*x1 - k0**2*x0 + x1))
        TL_temp = np.sum(vp*temp, axis=0)
        TL = -20*np.log10(np.abs(1+TL_temp))
 
        return TL
    
    
class ReflectionLining(Linings):
    
    '''
    Class calculates kz and Z for a reflection silencer.
    '''
    
    # geometry
    height = tr.Float()
    
    @property
    def S(self):
        
        return self.depth*self.height
        
    
    # method calculate kz and Z
    def Zkz(self, medium, freq):
        
        kz = (2*np.pi*freq)/medium.c
        Z = (medium.rho0*medium.c*kz)/(self.S*kz)
        
        return (kz, Z)
       
       
class AbsorptionLining(Linings):
    
    '''
    Class calculates kz and Z for a absorption silencer.
    '''
    
    # geometry
    height = tr.Float()
    
    # thickness of absorber
    dw = tr.Float()
    
    # flow resistance
    Xi = tr.Float(default_value=6000) #Ns/m⁴
    
    @property
    def S(self):
        
        return self.depth*self.height
    
    
    # method calculate kz and Z
    def Zkz(self, medium, freq):
        
        # Anpassen nach Vorbild von ReflectionLining!!!
        pass
        
    
          
       
       
       
       
























# =============================================================================
# # =============================================================================
# # class PlateResonator(tr.HasTraits):
# #     '''
# #     Parent class for all types of plate resonators linings.
# #     '''
# #     # geometrie
# #     length = tr.Float()
# #     depth = tr.Float()
# #     
# #     # number of modes of the plate
# #     Lp = tr.Int()   # x-direction
# #     Mp = tr.Int()   # y-direction
# #     
# #     # number of modes of the cavity
# #     Lc = tr.Int()   # x-direction
# #     Mc = tr.Int()   # y-direction
# #     Nc = tr.Int()   # z-direction
# #     
# #     # transfer of configuration, e.g. StandardPlateResonator 
# #     arrangement = tr.Instance()
# # =============================================================================
# 
# 
# # =============================================================================
# # class DummyLining(tr.HasTraits):
# #     
# #     # geometrie
# #     length = tr.Float()
# #     depth = tr.Float()
# #     height = tr.Float()
# #     
# #     # medium
# #     medium = tr.Instance(Fluid)
# #     
# #     # thickness of absorber
# #     dw = tr.Float(default_value=0)
# #     
# #     # flow resistance
# #     Xi = tr.Float(default_value=6000)   # Ns/m⁴
# #        
# #     # schallharte Wand  / Reflexionsschalldämpfer 
# #     def reflection(self, freq):
# #         self.S = self.depth*self.height
# #         self.kz = (2*np.pi*freq)/self.medium.c
# #         self.Z = (self.medium.rho0*self.medium.c*self.kz)/(self.S*self.kz)
# #         
# #         return (self.kz, self.Z)
# #     
# #     # Absorption
# #     def absorption(self, freq):
# #         self.S = self.depth*self.height
# #         
# #         X = (self.medium.rho0*freq)/self.Xi
# #         k = (2*np.pi*freq)/self.medium.c
# #         Z0 = self.medium.rho0*self.medium.c
# #         
# #         # nach Miki
# #         ka = (1+0.0109*X**(-0.618)-1j*0.160*X**(-0.618))*k
# #         Za = (1+0.070*X**(-0.632)-1j*0.107*X**(-0.632))*Z0
# #         
# #         self.Zw = -1j*Za*(1/np.tan(ka*self.dw))
# #         
# #         
# #         self.kz = np.sqrt(k**2-((1j*Z0)/(self.Zw*self.height)))
# #         self.Z = (Z0*k)/(self.S*self.kz)
# #         
# #         return (self.kz, self.Z)
# # =============================================================================
# 
# class DummyLining(tr.HasTraits):
#     
#     # geometrie
#     length = tr.Float()
#     depth = tr.Float()
#     height = tr.Float()
#     
#     # medium
#     medium = tr.Instance(Fluid)
#     
#     @property
#     def S(self):
#         
#         S = self.depth*self.height
#         
#         return S
# 
# class DummyReflection(DummyLining):
#     
#     def Zkz(self, freq):
#         
#         kz = (2*np.pi*freq)/self.medium.c
#         Z = (self.medium.rho0*self.medium.c*kz)/(self.S*kz)
#         
#         return (kz, Z)
#     
# # =============================================================================
# #     def kz(self, freq):
# #         
# #         kz = (2*np.pi*freq)/self.medium.c
# #         
# #         return kz
# #     
# #     def Z(self, freq):
# #         
# #         Z = (self.medium.rho0*self.medium.c*self.kz(freq))/(self.S*self.kz(freq))
# #         
# #         return Z 
# # =============================================================================
#         
#     
# # =============================================================================
# # class DummyReflection(DummyLining):
# #     
# #     def koeff(self, freq):
# #         self.S = self.depth*self.height
# #         self.kz = (2*np.pi*freq)/self.medium.c
# #         self.Z = (self.medium.rho0*self.medium.c*self.kz)/(self.S*self.kz)
# #         
# #         return (self.kz, self.Z)
# # =============================================================================
# 
# class DummyAbsorption(DummyLining):
#     
#     # thickness of absorber
#     dw = tr.Float()
#     
#     # flow resistance
#     Xi = tr.Float(default_value=6000) #Ns/m⁴
#     
#     def koeff(self, freq):
#         
#         
#          self.S = self.depth*self.height
#          
#          X = (self.medium.rho0*freq)/self.Xi
#          k = (2*np.pi*freq)/self.medium.c
#          Z0 = self.medium.rho0*self.medium.c
#          
#          # nach Miki
#          ka = (1+0.0109*X**(-0.618)-1j*0.160*X**(-0.618))*k
#          Za = (1+0.070*X**(-0.632)-1j*0.107*X**(-0.632))*Z0
#          
#          self.Zw = -1j*Za*(1/np.tan(ka*self.dw))
#          
#          
#          self.kz = np.sqrt(k**2-((1j*Z0)/(self.Zw*self.height)))
#          self.Z = (Z0*k)/(self.S*self.kz)
#          
#          return (self.kz, self.Z)
#     
#         
# #%% Plate Resonator classes
# # subclasses with different configurations
# 
# class PlateResonators(tr.HasTraits):
#     
#     '''
#     Parent class for different plate resonator configurations.
#     '''
#     
#     # geometry
#     length = tr.Float()
#     depth = tr.Float(default_value=0)
#     
#     # plate
#     plate = tr.Instance(Plate)
#     
#     # medium
#     medium = tr.Instance(Fluid)
#     
#     # cavity
#     cavity = tr.Instance(Cavity)
#     
#     # plate modes
#     J = tr.Instance(np.ndarray)
#     L = tr.Instance(np.ndarray)
#     
#     # flow -> eigene Klasse erstellen?
#     M = tr.Float(default_value=0)
#     
#    
# class SinglePlate(PlateResonators):
#     
#     '''
#     2D single plate resonator without flow
#     '''
# 
#     def zmatrix(self, freq):
#         
#         # cavity impedance matrix
#         Zc = self.cavity.ModImp(self.J, self.L, freq)
#         
#         # impedance matrix of plate induced sound
#         Zprad = np.zeros((len(self.J), len(self.L), len(freq)), dtype=complex)
#         
#         # cicular frequency and wave number
#         omega = 2*np.pi*freq
#         k0 = omega/self.medium.c
#         
#         for j in self.J:
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
#                     
#         Z = Zc+Zprad
#         
#         return Z
# =============================================================================
         

    
    