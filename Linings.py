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

#%%

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
        x0=Krm*M + k0
        x1=numpy.pi*L
        x2=Krm*self.length
        x3=-x1
        x4=1j*M
        x5=numpy.pi**4*L**4*x4
        x6=1j*k0
        x7=self.length**4*x6
        x8=numpy.pi**2*L**2
        x9=2*x8
        x10=self.length*x9
        x11=k0*x10
        x12=M*x9
        x13=self.length**2*x8
        x14=x13*x6
        x15=x13*x4
        x16=(-1)**L*x10
        x17=(1/2)*self.length/k0
        x18=Krp*self.length
        x19=Krp*M
        
        Zpll_temp = (1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltar)*(x0*x17*(-Krm**3*x7 - Krm**2*x15 + Krm*x14 - x0*x16*numpy.exp(-1j*x2) + x11 + x12*x2 + x5)/((x1 + x2)**2*(x2 + x3)**2) + x17*(k0 - x19)*(-Krp**3*x7 + Krp**2*x15 + Krp*x14 + x11 - x12*x18 + x16*(-k0 + x19)*numpy.exp(-1j*x18) - x5)/((x1 + x18)**2*(x18 + x3)**2))/(height_d*numpy.sqrt(-k0**2 + self.kappar(height_d)**2*(1 - M**2), dtype=complex))
       
        Zpll = np.sum(Zpll_temp*(np.identity(len(self.l))[:, :, np.newaxis, np.newaxis]), axis=2)
        
        
        # for j != l#
        x0=numpy.pi*J
        x1=Krm*self.length
        x2=numpy.pi*L
        x3=-x0
        x4=-x2
        x5=numpy.pi**2
        x6=L**2
        x7=x5*x6
        x8=self.length**2
        x9=Krm**2*x8
        x10=J + L
        x11=(-1)**x10
        x12=(-1)**(x10 + 1)
        x13=J**2
        x14=x13*x5
        x15=x5*(x13 - x6)
        #x16=J*L*x8/(k0*x10*(J - L))
        x16 = np.divide(J*L*x8, (k0*x10*(J - L)), np.zeros_like(J*L*k0, dtype=complex), where=(k0*x10*(J - L))!=0)
        x17=Krp*self.length
        x18=Krp*M
        x19=Krp**2*x8
        
        Zplj_temp = (1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltar)*(-x16*(-k0 + x18)*(k0 - x18)*((-1)**(L + 1)*x15*numpy.exp(-1j*x17) + x11*x19 + x12*x7 + x14 - x19)/((x0 + x17)*(x17 + x2)*(x17 + x3)*(x17 + x4)) - x16*(Krm*M + k0)**2*((-1)**J*x15*numpy.exp(-1j*x1) + x11*x9 + x12*x14 + x7 - x9)/((x0 + x1)*(x1 + x2)*(x1 + x3)*(x1 + x4)))/(height_d*numpy.sqrt(-k0**2 + self.kappar(height_d)**2*(1 - M**2), dtype=complex))

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
    
class SimpleTwoSidedPlateResonator(PlateResonators):
    
    '''
    2D two-sides plate resonator with similar plates.
    '''
    
    # plate on both sides
    plate = tr.Instance(Plate)
    
    # cavity on both sides
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
        x0=Krm*M + k0
        x1=numpy.pi*L
        x2=Krm*self.length
        x3=-x1
        x4=1j*M
        x5=numpy.pi**4*L**4*x4
        x6=1j*k0
        x7=self.length**4*x6
        x8=numpy.pi**2*L**2
        x9=2*x8
        x10=self.length*x9
        x11=k0*x10
        x12=M*x9
        x13=self.length**2*x8
        x14=x13*x6
        x15=x13*x4
        x16=(-1)**L*x10
        x17=(1/2)*self.length/k0
        x18=Krp*self.length
        x19=Krp*M
        
        # calculate Zpll_temp with additional factor for two-sided plate resonator
        Zpll_temp = (1+(-1)**R)*((1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltar)*(x0*x17*(-Krm**3*x7 - Krm**2*x15 + Krm*x14 - x0*x16*numpy.exp(-1j*x2) + x11 + x12*x2 + x5)/((x1 + x2)**2*(x2 + x3)**2) + x17*(k0 - x19)*(-Krp**3*x7 + Krp**2*x15 + Krp*x14 + x11 - x12*x18 + x16*(-k0 + x19)*numpy.exp(-1j*x18) - x5)/((x1 + x18)**2*(x18 + x3)**2))/(height_d*numpy.sqrt(-k0**2 + self.kappar(height_d)**2*(1 - M**2), dtype=complex)))
       
        Zpll = np.sum(Zpll_temp*(np.identity(len(self.l))[:, :, np.newaxis, np.newaxis]), axis=2)
        
        
        # for j != l#
        x0=numpy.pi*J
        x1=Krm*self.length
        x2=numpy.pi*L
        x3=-x0
        x4=-x2
        x5=numpy.pi**2
        x6=L**2
        x7=x5*x6
        x8=self.length**2
        x9=Krm**2*x8
        x10=J + L
        x11=(-1)**x10
        x12=(-1)**(x10 + 1)
        x13=J**2
        x14=x13*x5
        x15=x5*(x13 - x6)
        #x16=J*L*x8/(k0*x10*(J - L))
        x16 = np.divide(J*L*x8, (k0*x10*(J - L)), np.zeros_like(J*L*k0, dtype=complex), where=(k0*x10*(J - L))!=0)
        x17=Krp*self.length
        x18=Krp*M
        x19=Krp**2*x8
        
        # calculate Zplj_temp with additional factor for two-sided plate resonator
        Zplj_temp = (1+(-1)**R)*((1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltar)*(-x16*(-k0 + x18)*(k0 - x18)*((-1)**(L + 1)*x15*numpy.exp(-1j*x17) + x11*x19 + x12*x7 + x14 - x19)/((x0 + x17)*(x17 + x2)*(x17 + x3)*(x17 + x4)) - x16*(Krm*M + k0)**2*((-1)**J*x15*numpy.exp(-1j*x1) + x11*x9 + x12*x14 + x7 - x9)/((x0 + x1)*(x1 + x2)*(x1 + x3)*(x1 + x4)))/(height_d*numpy.sqrt(-k0**2 + self.kappar(height_d)**2*(1 - M**2), dtype=complex)))

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

        # calculate TL_temp with additional factor for two-sided plate resonator
        TL_temp = np.sum(vp*(2*temp), axis=0)
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

         

    
    