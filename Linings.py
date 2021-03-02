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
    
    # method to calculate kappar
    def kappar(self, height_d):
        
        R = self.cavity.r[np.newaxis, np.newaxis, :, np.newaxis]
        
        return ((R*np.pi)/(height_d))
        
    
    # methode calculate the impedance matrix of the plate resonator
    def zmatrix(self, height_d, M, freq):
        
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
        Krp = (-k0*M-1j*np.sqrt((1-M**2)*self.kappar(height_d)**2-k0**2, dtype=complex))/(1-M**2)
        Krm = (k0*M-1j*np.sqrt((1-M**2)*self.kappar(height_d)**2-k0**2, dtype=complex))/(1-M**2)
        
        # for j=l
        x0=Krp*M
        x1=numpy.pi**2*L**2
        x2=2*x1
        x3=-x2
        x4=self.length**2
        x5=Krp**2*x4
        x6=2*x5
        x7=1j*self.length**3
        x8=-k0*x2
        x9=1j*self.length
        x10=M*x1*x9
        x11=k0*x1
        x12=(-k0*x5 + x11)**(-1.0)
        x13=x12/(x1 - x5)
        x14=x11*x4
        x15=x1*x4
        x16=(-1)**L
        x17=numpy.exp(Krp*x9)
        x18=x12/(x1*x17 - x17*x5)
        x19=x15*x16
        x20=Krm*M
        x21=Krm**2*x4
        x22=2*x21
        x23=numpy.exp(Krm*x9)
        x24=x1*x23
        x25=1/((-x1 + x21)*(-x21*x23 + x24))
        x26=x19*x25
        x27=x24*x25*x4
        x28=x20/k0
        
        Zpll_temp = (1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltar)*((k0 - x0)*(-Krp*x7/(x3 + x6) - x0*x13*x15 + x0*x18*x19 + x10/(k0*x6 + x8) + x13*x14 - x14*x16*x18) + (k0 + x20)*(-Krm*x7/(x22 + x3) - x10/(k0*x22 + x8) + x26*x28 + x26 - x27*x28 - x27))/(height_d*numpy.sqrt(-k0**2 + self.kappar(height_d)**2*(1 - M)**2, dtype=complex))
        
        Zpll = np.sum(Zpll_temp*(np.identity(len(self.l))[:, :, np.newaxis, np.newaxis]), axis=2)
        
        # for j != l
        x0=Krp*M
        x1=J**2
        x2=L**2
        x3=self.length**2
        x4=J*L*x3
        #x5=numpy.pi*x4/(numpy.pi*x1 - numpy.pi*x2)
        x5 = np.divide(numpy.pi*x4, (numpy.pi*x1 - numpy.pi*x2), out=np.zeros_like(J*L, dtype=float), where=(numpy.pi*x1 - numpy.pi*x2)!=0)
        x6=numpy.pi**2
        x7=x1*x6
        x8=Krp**2*x3
        x9=(k0*x7 - k0*x8)**(-1.0)
        x10=k0*x9
        x11=x10*x5
        x12=x0*x9
        x13=x12*x5
        x14=x2*x6
        x15=x4*x6
        x16=x15/(x14 - x8)
        x17=(-1)**J
        x18=(-1)**L
        x19=x17*x18
        x20=1j*self.length
        x21=numpy.exp(Krp*x20)
        x22=x15*x18/(x14*x21 - x21*x8)
        x23=Krm*M
        x24=Krm**2*x3
        x25=x5/(-x24 + x7)
        x26=x19*x25
        x27=x23/k0
        x28=numpy.exp(Krm*x20)
        x29=x15*x17/((x24 - x7)*(x14*x28 - x24*x28))
        x30=x18*x28*x29
        
        Zplj_temp = (1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltar)*((k0 - x0)*(x10*x16 - x10*x22 - x11*x19 + x11 - x12*x16 + x12*x22 + x13*x19 - x13) + (k0 + x23)*(-x25*x27 - x25 + x26*x27 + x26 + x27*x29 - x27*x30 + x29 - x30))/(height_d*numpy.sqrt(-k0**2 + self.kappar(height_d)**2*(1 - M)**2, dtype=complex))

        Zplj = np.sum(Zplj_temp, axis=2)
        
        # calculate the overall impedance matrix of the plate induced sound
        Zprad = Zplj+Zpll
        
        # calculate the total impedance matrix of the lining
        Z = Zc+Zprad
        
        return Z
   
    # method calculate the plate velocity
    def platevelocity(self, height_d, I, M, freq):
        
        # impedance matrix
        Z = self.zmatrix(height_d, M, freq)
        
        # L matrix with material properties of the plate
        Lmatrix = self.plate.lmatrix(self.length, self.depth, self.l, freq)
        
        # solving the linear system of equations
        # building lhs matrix from Z and Lmatrix
        lhs = Z+Lmatrix
        
        # calculate the plate velocity
        vp = np.zeros_like(I, dtype=complex)
        for i in range(len(freq)):
            
            vp[:,i] = np.linalg.solve(lhs[:,:,i], -I[:,i])
        
        return vp
        
    # method calculates the transmission loss of the plate silencer
    def transmissionloss(self, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis]
        
        # plate velocity
        vp = self.platevelocity(height_d, I, M, freq)
        
        # calculate transmission loss
        x0=(M + 1)**(-1.0)
        x1=numpy.pi**2*J**2*k0
        x2=self.length*k0
        x3=numpy.pi*J*x2/(-self.length**2*k0**3 + M**2*x1 + 2*M*x1 + x1)
        x4=M*x3
        x5=(-1)**J*numpy.exp(1j*x0*x2)
        
        temp = (1/2)*x0*(-x3*x5 + x3 - x4*x5 + x4)/(self.cavity.medium.c*height_d)

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
         

    
    