#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:58:19 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
import numba as nb

from .Fluid import Fluid
from .Cavity import Cavities2D, Cavities3D
from .Plate import Plate, Plate3D


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
    # 2D
    j = tr.Instance(np.ndarray)
    l = tr.Instance(np.ndarray)
    # 3D
    k = tr.Instance(np.ndarray)
    n = tr.Instance(np.ndarray)
    
    # duct modes
    # z-axis
    t = tr.Instance(np.ndarray)

class SinglePlateResonator(PlateResonators):
    
    '''
    2D single plate resonatr without flow.
    '''
    
    # plate
    plate = tr.Instance(Plate)
    
    # cavity
    cavity = tr.Instance(Cavities2D)
    
    # property method to define the Kronecker delta
    @property
    def deltat(self):
        
        return np.eye(len(self.t),1)[np.newaxis, np.newaxis, :]
    
    # method to calculate kappat
    def kappat(self, height_d):
        
        T = self.t[np.newaxis, np.newaxis, :, np.newaxis]
        
        return ((T*np.pi)/(height_d))
        
    
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
        T = self.t[np.newaxis, np.newaxis, :, np.newaxis]
        
        # calculate the impedance matrix of the plate induced sound
        Ktp = (-k0*M-1j*np.sqrt((1-M**2)*self.kappat(height_d)**2-k0**2, dtype=complex))/(1-M**2)
        Ktm = (k0*M-1j*np.sqrt((1-M**2)*self.kappat(height_d)**2-k0**2, dtype=complex))/(1-M**2)
        
        # for j=l
        x0=Ktm*M + k0
        x1=numpy.pi*L
        x2=Ktm*self.length
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
        x18=Ktp*self.length
        x19=Ktp*M
        
        Zpll_temp = (1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltat)*(x0*x17*(-Ktm**3*x7 - Ktm**2*x15 + Ktm*x14 - x0*x16*numpy.exp(-1j*x2) + x11 + x12*x2 + x5)/((x1 + x2)**2*(x2 + x3)**2) + x17*(k0 - x19)*(-Ktp**3*x7 + Ktp**2*x15 + Ktp*x14 + x11 - x12*x18 + x16*(-k0 + x19)*numpy.exp(-1j*x18) - x5)/((x1 + x18)**2*(x18 + x3)**2))/(height_d*numpy.sqrt(-k0**2 + self.kappat(height_d)**2*(1 - M**2), dtype=complex))
       
        Zpll = np.sum(Zpll_temp*(np.identity(len(self.l))[:, :, np.newaxis, np.newaxis]), axis=2)
        
        
        # for j != l#
        x0=numpy.pi*J
        x1=Ktm*self.length
        x2=numpy.pi*L
        x3=-x0
        x4=-x2
        x5=numpy.pi**2
        x6=L**2
        x7=x5*x6
        x8=self.length**2
        x9=Ktm**2*x8
        x10=J + L
        x11=(-1)**x10
        x12=(-1)**(x10 + 1)
        x13=J**2
        x14=x13*x5
        x15=x5*(x13 - x6)
        #x16=J*L*x8/(k0*x10*(J - L))
        x16 = np.divide(J*L*x8, (k0*x10*(J - L)), np.zeros_like(J*L*k0, dtype=complex), where=(k0*x10*(J - L))!=0)
        x17=Ktp*self.length
        x18=Ktp*M
        x19=Ktp**2*x8
        
        Zplj_temp = (1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltat)*(-x16*(-k0 + x18)*(k0 - x18)*((-1)**(L + 1)*x15*numpy.exp(-1j*x17) + x11*x19 + x12*x7 + x14 - x19)/((x0 + x17)*(x17 + x2)*(x17 + x3)*(x17 + x4)) - x16*(Ktm*M + k0)**2*((-1)**J*x15*numpy.exp(-1j*x1) + x11*x9 + x12*x14 + x7 - x9)/((x0 + x1)*(x1 + x2)*(x1 + x3)*(x1 + x4)))/(height_d*numpy.sqrt(-k0**2 + self.kappat(height_d)**2*(1 - M**2), dtype=complex))

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
    
    # method calculates the transmission factor of the plate silencer
    def transmissionfactor(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis]
        
        # calculate transmission loss
        x0=(M + 1)**(-1.0)
        x1=numpy.pi**2*J**2*k0
        x2=self.length*k0
        x3=numpy.pi*J*x2/(-self.length**2*k0**3 + M**2*x1 + 2*M*x1 + x1)
        x4=M*x3
        x5=(-1)**J*numpy.exp(1j*x0*x2)
        
        temp = (1/2)*x0*(-x3*x5 + x3 - x4*x5 + x4)/(self.cavity.medium.c*height_d)
        
        tra_fac = np.exp(-1j*((k0*self.length)/(1+M)))*(np.sum(vp*temp, axis=0)+1)
        
        return tra_fac
    
    # method calculates the transmittance of the plate silencer
    def transmittance(self, vp, height_d, I, M, medium, freq):
        
        # call transmission factor
        tra_fac = self.transmissionfactor(vp, height_d, I, M, medium, freq)
        
        # calculate transmittance
        tra = np.abs(tra_fac)**2
        
        return tra
    
    # method calculates the reflection factor of the plate silencer
    def reflectionfactor(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis]
        
        # calculate reflactance
        x0=1j*self.length*k0/(M - 1)
        x1=numpy.pi**2*J**2
        
        temp = (1/2)*numpy.pi*self.length*J*((-1)**J - numpy.exp(-x0))*numpy.exp(x0)/(self.cavity.medium.c*height_d*(self.length**2*k0**2 - M**2*x1 + 2*M*x1 - x1))
        
        ref_fac = np.sum(vp*temp, axis=0)
        
        return ref_fac
    
    # method calculates the reflectance of the plate silencer
    def reflectance(self, vp, height_d, I, M, medium, freq):
        
        # call reflection factor
        ref_fac = self.reflectionfactor(vp, height_d, I, M, medium, freq)
        
        # calculate reflectance with energetic correction due to mean flow
        ref = np.abs(ref_fac)**2*((1-M)/(1+M))**2
        
        return ref
    
    # method calculates the dissipation of the plate silencer
    def dissipation(self, vp, height_d, I, M, medium, freq):
        
        # call transmittance
        tra = self.transmittance(vp, height_d, I, M, medium, freq)
        
        # call reflectance
        ref = self.reflectance(vp, height_d, I, M, medium, freq)
        
        # calculate dissipation
        dis = 1-tra-ref
        
        return dis
    
    # method calculates the transmission loss of the plate silencer
    def transmissionloss(self, vp, height_d, I, M, medium, freq):
        
        # transmittance
        tra = self.transmittance(vp, height_d, I, M, medium, freq)
        
        # transmission loss
        TL = -10*np.log10(tra)
        
        return TL
    
class SimpleTwoSidedPlateResonator(PlateResonators):
    
    '''
    2D two-sides plate resonator with similar plates.
    '''
    
    # plate on both sides
    plate = tr.Instance(Plate)
    
    # cavity on both sides
    cavity = tr.Instance(Cavities2D)
    
    # property method to define the Kronecker delta
    @property
    def deltat(self):
        
        return np.eye(len(self.t),1)[np.newaxis, np.newaxis, :]
    
    # method to calculate kappat
    def kappat(self, height_d):
        
        T = self.t[np.newaxis, np.newaxis, :, np.newaxis]
        
        return ((T*np.pi)/(height_d))
    
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
        T = self.t[np.newaxis, np.newaxis, :, np.newaxis]
        
        # calculate the impedance matrix of the plate induced sound
        Ktp = (-k0*M-1j*np.sqrt((1-M**2)*self.kappat(height_d)**2-k0**2, dtype=complex))/(1-M**2)
        Ktm = (k0*M-1j*np.sqrt((1-M**2)*self.kappat(height_d)**2-k0**2, dtype=complex))/(1-M**2)
        
        # for j=l
        x0=Ktm*M + k0
        x1=numpy.pi*L
        x2=Ktm*self.length
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
        x18=Ktp*self.length
        x19=Ktp*M
        
        # calculate Zpll_temp with additional factor for two-sided plate resonator
        Zpll_temp = (1+(-1)**T)*((1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltat)*(x0*x17*(-Ktm**3*x7 - Ktm**2*x15 + Ktm*x14 - x0*x16*numpy.exp(-1j*x2) + x11 + x12*x2 + x5)/((x1 + x2)**2*(x2 + x3)**2) + x17*(k0 - x19)*(-Ktp**3*x7 + Ktp**2*x15 + Ktp*x14 + x11 - x12*x18 + x16*(-k0 + x19)*numpy.exp(-1j*x18) - x5)/((x1 + x18)**2*(x18 + x3)**2))/(height_d*numpy.sqrt(-k0**2 + self.kappat(height_d)**2*(1 - M**2), dtype=complex)))
       
        Zpll = np.sum(Zpll_temp*(np.identity(len(self.l))[:, :, np.newaxis, np.newaxis]), axis=2)
        
        
        # for j != l#
        x0=numpy.pi*J
        x1=Ktm*self.length
        x2=numpy.pi*L
        x3=-x0
        x4=-x2
        x5=numpy.pi**2
        x6=L**2
        x7=x5*x6
        x8=self.length**2
        x9=Ktm**2*x8
        x10=J + L
        x11=(-1)**x10
        x12=(-1)**(x10 + 1)
        x13=J**2
        x14=x13*x5
        x15=x5*(x13 - x6)
        #x16=J*L*x8/(k0*x10*(J - L))
        x16 = np.divide(J*L*x8, (k0*x10*(J - L)), np.zeros_like(J*L*k0, dtype=complex), where=(k0*x10*(J - L))!=0)
        x17=Ktp*self.length
        x18=Ktp*M
        x19=Ktp**2*x8
        
        # calculate Zplj_temp with additional factor for two-sided plate resonator
        Zplj_temp = (1+(-1)**T)*((1/2)*1j*self.cavity.medium.c*self.cavity.medium.rho0*(2 - self.deltat)*(-x16*(-k0 + x18)*(k0 - x18)*((-1)**(L + 1)*x15*numpy.exp(-1j*x17) + x11*x19 + x12*x7 + x14 - x19)/((x0 + x17)*(x17 + x2)*(x17 + x3)*(x17 + x4)) - x16*(Ktm*M + k0)**2*((-1)**J*x15*numpy.exp(-1j*x1) + x11*x9 + x12*x14 + x7 - x9)/((x0 + x1)*(x1 + x2)*(x1 + x3)*(x1 + x4)))/(height_d*numpy.sqrt(-k0**2 + self.kappat(height_d)**2*(1 - M**2), dtype=complex)))

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
    
    # method calculates the transmission factor of the plate silencer
    def transmissionfactor(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis]
        
        # calculate transmission coefficient
        x0=(M + 1)**(-1.0)
        x1=numpy.pi**2*J**2*k0
        x2=self.length*k0
        x3=numpy.pi*J*x2/(-self.length**2*k0**3 + M**2*x1 + 2*M*x1 + x1)
        x4=M*x3
        x5=(-1)**J*numpy.exp(1j*x0*x2)
        
        temp = (1/2)*x0*(-x3*x5 + x3 - x4*x5 + x4)/(self.cavity.medium.c*height_d)
        
        tra_fac = np.exp(-1j*((k0*self.length)/(1+M)))*(np.sum(vp*(2*temp), axis=0)+1)
        
        return tra_fac
    
    # method calculates the transmittance of the plate silencer
    def transmittance(self, vp, height_d, I, M, medium, freq):
        
        # call transmission factor
        tra_fac = self.transmissionfactor(vp, height_d, I, M, medium, freq)
        
        # calculate transmittance
        tra = np.abs(tra_fac)**2
        
        return tra
    
    # method calculates the reflection factor of the plate silencer
    def reflectionfactor(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis]
        
        # calculate reflactance
        x0=1j*self.length*k0/(M - 1)
        x1=numpy.pi**2*J**2
        
        temp = (1/2)*numpy.pi*self.length*J*((-1)**J - numpy.exp(-x0))*numpy.exp(x0)/(self.cavity.medium.c*height_d*(self.length**2*k0**2 - M**2*x1 + 2*M*x1 - x1))
        
        ref_fac = np.sum(vp*(2*temp), axis=0)
        
        return ref_fac
    
    # method calculates the reflectance of the plate silencer
    def reflectance(self, vp, height_d, I, M, medium, freq):
        
        # call reflection factor
        ref_fac = self.reflectionfactor(vp, height_d, I, M, medium, freq)
        
        # calculate reflectance with energetic correction due to mean flow
        ref = np.abs(ref_fac)**2*((1-M)/(1+M))**2
        
        return ref
    
    # method calculates the dissipation of the plate silencer
    def dissipation(self, vp, height_d, I, M, medium, freq):
        
        # call transmittance
        tra = self.transmittance(vp, height_d, I, M, medium, freq)
        
        # call reflectance
        ref = self.reflectance(vp, height_d, I, M, medium, freq)
        
        # calculate dissipation
        dis = 1-tra-ref
        
        return dis
    
    # method calculates the transmission loss of the plate silencer
    def transmissionloss(self, vp, height_d, I, M, medium, freq):
        
        # transmission coefficient
        tra = self.transmittance(vp, height_d, I, M, medium, freq)        

        # transmission loss
        TL = -10*np.log10(tra)        

        return TL 
    

    
class ReflectionLining(Linings):
    
    '''
    Class calculates kz and Z for a reflection silencer.
    '''
    
    # geometry
    height = tr.Float()
    
    def S(self, height_d):
        
        return self.depth*(2*self.height+height_d)
        
    
    # method calculate kz and Z
    def Zkz(self, medium, height_d, freq):
        
        kz = (2*np.pi*freq)/medium.c
        Z = (medium.rho0*medium.c*kz)/(self.S(height_d)*kz)
        
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


#%% SinglePlateResonator3D
# method to calculate impedance matrix of plate induced sound for SinglePlateResonator3D
# accelerated by numba 
@nb.njit#(parallel=True)
def get_zprad(c, rho0, j, k, l, n, s, t, length, depth, height_d, M, freq):
    
    # circular frequency and wave number
    omega = 2*np.pi*freq
    k0 = omega/c
    
    # impedance matrix of plate induced sound
    # define empty impedance array
    Zprad = np.zeros((len(j), len(k), len(l), len(n), len(freq)), dtype=np.complex128)
    
    for J in j:
        
        for K in k:
                
                for L in l:
                    
                    for N in n:
                        
                        #define empty sum array
                        Sum = np.zeros(len(freq), dtype=np.complex128)
                        
                        for S in s:
                            
                            for T in t:
                                
                                # calculate the kronecker delta
                                deltas = 1 if S==0 else 0
                                deltat = 1 if T==0 else 0
                                
                                # calculate kappars
                                kappast = np.sqrt(((S*np.pi)/(depth))**2+((T*np.pi)/(height_d))**2)
                                
                                # 
                                Kstp = (-k0*M-1j*np.sqrt((1-M**2)*kappast**2-k0**2))/(1-M**2)
                                Kstm = (k0*M-1j*np.sqrt((1-M**2)*kappast**2-k0**2))/(1-M**2)
                                # j=l
                                if J==L:
                                    
                                    # j=l, k=s
                                    if K==S:
                                        
                                        # j=l, k=s, n=s
                                        if N==S:
                                            
                                            Sum += 0
                                        
                                        # j=l, k=s
                                        else:
                                            
                                            Sum += 0
                                            
                                    # j=l, n=s
                                    elif N==S:
                                        
                                        Sum += 0

                                    # j=l
                                    else:
                                        
                                        x0=Kstm*M + k0
                                        x1=numpy.pi*L
                                        x2=Kstm*length
                                        x3=-x1
                                        x4=1j*M
                                        x5=(np.pi)**(4+0j)*(L)**(4+0j)*x4
                                        x6=1j*k0
                                        x7=(length)**(4+0j)*x6
                                        x8=(np.pi)**(2+0j)
                                        x9=(L)**(2+0j)*x8
                                        x10=2*x9
                                        x11=length*x10
                                        x12=k0*x11
                                        x13=M*x10
                                        x14=(length)**(2+0j)*x9
                                        x15=x14*x6
                                        x16=x14*x4
                                        x17=(-1)**(L+0j)*x11
                                        x18=K + S
                                        x19=-S
                                        x20=N + S
                                        x21=(1/2)*length*(depth)**(2+0j)*K*N*((-1)**(K + N+0j) + (-1)**(x18 + 1+0j) + (-1)**(x20 + 1+0j) + 1)/(k0*x18*x20*x8*(K + x19)*(N + x19))
                                        x22=Kstp*length
                                        x23=Kstp*M
                                        
                                        Sum += (1/2)*1j*c*rho0*(2 - deltat)*(2 - deltas)*(x0*x21*(-(Kstm)**(3+0j)*x7 - (Kstm)**(2+0j)*x16 + Kstm*x15 - x0*x17*numpy.exp(-1j*x2) + x12 + x13*x2 + x5)/((x1 + x2)**(2+0j)*(x2 + x3)**(2+0j)) + x21*(k0 - x23)*(-(Kstp)**(3+0j)*x7 + (Kstp)**(2+0j)*x16 + Kstp*x15 + x12 - x13*x22 + x17*(-k0 + x23)*numpy.exp(-1j*x22) - x5)/((x1 + x22)**(2+0j)*(x22 + x3)**(2+0j)))/(depth*height_d*(kappast**2*(1 - M**2) - k0**2)**(1/2+0j))
                                        

                                # k=s
                                elif K==S:
                                        
                                    Sum += 0

                                # n=s
                                elif N==S:
                                    
                                    Sum += 0
                                        
                                # jk!=lm
                                else:
                                    
                                    x0=numpy.pi*J
                                    x1=Kstm*length
                                    x2=numpy.pi*L
                                    x3=-x0
                                    x4=-x2
                                    x5=(np.pi)**(2+0j)
                                    x6=(L)**(2+0j)
                                    x7=x5*x6
                                    x8=(length)**(2+0j)
                                    x9=(Kstm)**(2+0j)*x8
                                    x10=J + L
                                    x11=(-1)**(x10+0j)
                                    x12=(-1)**(x10 + 1+0j)
                                    x13=(J)**(2+0j)
                                    x14=x13*x5
                                    x15=x5*(x13 - x6)
                                    x16=K + S
                                    x17=-S
                                    x18=N + S
                                    x19=(depth)**(2+0j)*J*K*L*N*x8*((-1)**(K + N+0j) + (-1)**(x16 + 1+0j) + (-1)**(x18 + 1+0j) + 1)/(k0*x10*x16*x18*x5*(J - L)*(K + x17)*(N + x17))
                                    x20=Kstp*length
                                    x21=Kstp*M
                                    x22=(Kstp)**(2+0j)*x8
                                    
                                    Sum += (1/2)*1j*c*rho0*(2 - deltat)*(2 - deltas)*(-x19*(-k0 + x21)*(k0 - x21)*((-1)**(L + 1+0j)*x15*numpy.exp(-1j*x20) + x11*x22 + x12*x7 + x14 - x22)/((x0 + x20)*(x2 + x20)*(x20 + x3)*(x20 + x4)) - x19*(Kstm*M + k0)**(2+0j)*((-1)**(J+0j)*x15*numpy.exp(-1j*x1) + x11*x9 + x12*x14 + x7 - x9)/((x0 + x1)*(x1 + x2)*(x1 + x3)*(x1 + x4)))/(depth*height_d*(kappast**2*(1 - M**2) - k0**2)**(1/2+0j))
                                
                        Zprad[J-1, K-1, L-1, N-1,:] = Sum
    
    return Zprad

        
class SinglePlateResonator3D(PlateResonators):
    
    '''
    3D single plate silencer.
    '''
    
    #k = tr.Instance(np.ndarray)
    #n = tr.Instance(np.ndarray)
    
    # plate
    plate = tr.Instance(Plate3D)
    
    # cavity
    cavity = tr.Instance(Cavities3D)
    
    # methode calculate the impedance matrix of the plate resonator 
    # accelerated by numba
    def zmatrix(self, height_d, M, freq):
        
        # cavity impedance matrix
        Zc = self.cavity.cavityimpedance(self.length, self.depth, self.j, self.k, self.l, self.n, freq)
        
        # calculate impedance matrix of plate induced sound
        # accelerated by numba
        Zprad = get_zprad(self.cavity.medium.c, self.cavity.medium.rho0, self.j, self.k, self.l, self.n, self.cavity.s, self.t, self.length, self.depth, height_d, M, freq)
        
        # calculate the total impedance matrix of the lining
        Z = Zc+Zprad
        
        return Z
    
    # method calculate the plate velocity
    def platevelocity(self, height_d, I, M, freq):
        
        # impedance matrix
        Z = self.zmatrix(height_d, M, freq)
        
        # L matrix with material properties of the plate
        Lmatrix = self.plate.lmatrix(self.length, self.depth, self.l, self.n, freq)
        
        # solving the linear system of equations
        # building lhs matrix from Z and Lmatrix
        lhs = Z+Lmatrix
        
        #calculate the plate velocity
        vp = np.zeros_like(I, dtype = complex)
        for i in range(len(freq)):
            
            vp[:,:,i] = np.linalg.tensorsolve(lhs[:,:,:,:,i], -I[:,:,i], axes=(0,1))
        
        return vp
    
    # method calculates the transmission factor of the plate silencer
    def transmissionfactor(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended arrays of modes
        J = self.j[:, np.newaxis, np.newaxis]
        K = self.k[np.newaxis, :, np.newaxis]
        
        # calculate transmission coefficient
        x0=(np.pi)**(2+0j)*(J)**(2+0j)
        x1=self.length*k0/(M + 1) + numpy.pi*J
        
        temp = (1/2)*self.length*J*((-1)**(K + 1+0j) - numpy.exp(1j*x1) + numpy.exp(1j*(numpy.pi*K + x1)) + 1)/(medium.c*height_d*K*(-(self.length)**(2+0j)*(k0)**(2+0j) + (M)**(2+0j)*x0 + 2*M*x0 + x0))
        
        tra_fac = np.exp(-1j*((k0*self.length)/(1+M)))*(np.sum(vp*temp, axis=(0,1))+1)
        
        return tra_fac
    
    # method calculates the transmittance of the plate silencer
    def transmittance(self, vp, height_d, I, M, medium, freq):
        
        # call transmission factor
        tra_fac = self.transmissionfactor(vp, height_d, I, M, medium, freq)
        
        # calculation of transmittance
        tra = np.abs(tra_fac)**2
        
        return tra
    
    # method calculates the reflection factor of the plate silencer
    def reflectionfactor(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis, np.newaxis]
        K = self.k[np.newaxis, :, np.newaxis]
        
        # calculate reflection coefficient
        x0=self.length*k0/(M - 1)
        x1=1j*x0
        x2=numpy.pi**2*J**2
        
        temp = (1/2)*self.length*J*((-1)**J + (-1)**(J + K + 1) + numpy.exp(-1j*(-numpy.pi*K + x0)) - numpy.exp(-x1))*numpy.exp(x1)/(medium.c*height_d*K*(self.length**2*k0**2 - M**2*x2 + 2*M*x2 - x2))
        
        ref_fac = np.sum(vp*temp, axis=(0,1))
        
        return ref_fac
    
    # method calculates the reflectance of the plate silencer
    def reflectance(self, vp, height_d, I, M, medium, freq):
        
        # call reflection factor
        ref_fac = self.reflectionfactor(vp, height_d, I, M, medium, freq)
        
        # calculate reflectance with energetic correction due to mean flow
        ref = np.abs(ref_fac)**2*((1-M)/(1+M))**2
        
        return ref
    
    # method calculates the dissipation of the plate silencer
    def dissipation(self, vp, height_d, I, M, medium, freq):
        
        # call transmittance
        tra = self.transmittance(vp, height_d, I, M, medium, freq)
        
        # call reflectance
        ref = self.reflectance(vp, height_d, I, M, medium, freq)
        
        # calculate dissipation
        dis = 1-tra-ref
        
        return dis
    
    # method calculate the transmission loss of the plate silencer
    def transmissionloss(self, vp, height_d, I, M, medium, freq):
        
        # transmission coefficient
        tra = self.transmittance(vp, height_d, I, M, medium, freq)
        
        # transmission loss
        TL = -10*np.log10(tra)
        
        return TL
    
#%% SimpleTwoSidedPlateResonator3D

# method to calculate impedance matrix of plate induced sound for SinglePlateResonator3D
# accelerated by numba 
@nb.njit#(parallel=True)
def get_zprad_twosided(c, rho0, j, k, l, n, s, t, length, depth, height_d, M, freq):
    
    # circular frequency and wave number
    omega = 2*np.pi*freq
    k0 = omega/c
    
    # impedance matrix of plate induced sound
    # define empty impedance array
    Zprad = np.zeros((len(j), len(k), len(l), len(n), len(freq)), dtype=np.complex128)
    
    for J in j:
        
        for K in k:
                
                for L in l:
                    
                    for N in n:
                        
                        #define empty sum array
                        Sum = np.zeros(len(freq), dtype=np.complex128)
                        
                        for S in s:
                            
                            for T in t:
                                
                                # calculate the kronecker delta
                                deltas = 1 if S==0 else 0
                                deltat = 1 if T==0 else 0
                                
                                # calculate kappars
                                kappast = np.sqrt(((S*np.pi)/(depth))**2+((T*np.pi)/(height_d))**2)
                                
                                # 
                                Kstp = (-k0*M-1j*np.sqrt((1-M**2)*kappast**2-k0**2))/(1-M**2)
                                Kstm = (k0*M-1j*np.sqrt((1-M**2)*kappast**2-k0**2))/(1-M**2)
                                # j=l
                                if J==L:
                                    
                                    # j=l, k=s
                                    if K==S:
                                        
                                        # j=l, k=s, n=s
                                        if N==S:
                                            
                                            Sum += 0
                                        
                                        # j=l, k=s
                                        else:
                                            
                                            Sum += 0
                                            
                                    # j=l, n=s
                                    elif N==S:
                                        
                                        Sum += 0

                                    # j=l
                                    else:
                                        
                                        x0=Kstm*M + k0
                                        x1=numpy.pi*L
                                        x2=Kstm*length
                                        x3=-x1
                                        x4=1j*M
                                        x5=(np.pi)**(4+0j)*(L)**(4+0j)*x4
                                        x6=1j*k0
                                        x7=(length)**(4+0j)*x6
                                        x8=(np.pi)**(2+0j)
                                        x9=(L)**(2+0j)*x8
                                        x10=2*x9
                                        x11=length*x10
                                        x12=k0*x11
                                        x13=M*x10
                                        x14=(length)**(2+0j)*x9
                                        x15=x14*x6
                                        x16=x14*x4
                                        x17=(-1)**(L+0j)*x11
                                        x18=K + S
                                        x19=-S
                                        x20=N + S
                                        x21=(1/2)*length*(depth)**(2+0j)*K*N*((-1)**(K + N+0j) + (-1)**(x18 + 1+0j) + (-1)**(x20 + 1+0j) + 1)/(k0*x18*x20*x8*(K + x19)*(N + x19))
                                        x22=Kstp*length
                                        x23=Kstp*M
                                        
                                        Sum += (1+(-1)**T)*(1/2)*1j*c*rho0*(2 - deltat)*(2 - deltas)*(x0*x21*(-(Kstm)**(3+0j)*x7 - (Kstm)**(2+0j)*x16 + Kstm*x15 - x0*x17*numpy.exp(-1j*x2) + x12 + x13*x2 + x5)/((x1 + x2)**(2+0j)*(x2 + x3)**(2+0j)) + x21*(k0 - x23)*(-(Kstp)**(3+0j)*x7 + (Kstp)**(2+0j)*x16 + Kstp*x15 + x12 - x13*x22 + x17*(-k0 + x23)*numpy.exp(-1j*x22) - x5)/((x1 + x22)**(2+0j)*(x22 + x3)**(2+0j)))/(depth*height_d*(kappast**2*(1 - M**2) - k0**2)**(1/2+0j))
                                        

                                # k=s
                                elif K==S:
                                        
                                    Sum += 0

                                # n=s
                                elif N==S:
                                    
                                    Sum += 0
                                        
                                # jk!=lm
                                else:
                                    
                                    x0=numpy.pi*J
                                    x1=Kstm*length
                                    x2=numpy.pi*L
                                    x3=-x0
                                    x4=-x2
                                    x5=(np.pi)**(2+0j)
                                    x6=(L)**(2+0j)
                                    x7=x5*x6
                                    x8=(length)**(2+0j)
                                    x9=(Kstm)**(2+0j)*x8
                                    x10=J + L
                                    x11=(-1)**(x10+0j)
                                    x12=(-1)**(x10 + 1+0j)
                                    x13=(J)**(2+0j)
                                    x14=x13*x5
                                    x15=x5*(x13 - x6)
                                    x16=K + S
                                    x17=-S
                                    x18=N + S
                                    x19=(depth)**(2+0j)*J*K*L*N*x8*((-1)**(K + N+0j) + (-1)**(x16 + 1+0j) + (-1)**(x18 + 1+0j) + 1)/(k0*x10*x16*x18*x5*(J - L)*(K + x17)*(N + x17))
                                    x20=Kstp*length
                                    x21=Kstp*M
                                    x22=(Kstp)**(2+0j)*x8
                                    
                                    Sum += (1+(-1)**T)*(1/2)*1j*c*rho0*(2 - deltat)*(2 - deltas)*(-x19*(-k0 + x21)*(k0 - x21)*((-1)**(L + 1+0j)*x15*numpy.exp(-1j*x20) + x11*x22 + x12*x7 + x14 - x22)/((x0 + x20)*(x2 + x20)*(x20 + x3)*(x20 + x4)) - x19*(Kstm*M + k0)**(2+0j)*((-1)**(J+0j)*x15*numpy.exp(-1j*x1) + x11*x9 + x12*x14 + x7 - x9)/((x0 + x1)*(x1 + x2)*(x1 + x3)*(x1 + x4)))/(depth*height_d*(kappast**2*(1 - M**2) - k0**2)**(1/2+0j))
                                
                        Zprad[J-1, K-1, L-1, N-1,:] = Sum
    
    return Zprad

    
class SimpleTwoSidedPlateResonator3D(PlateResonators):
    
    '''
    3D two-sided plate silencer with similar plates.
    '''
    
    # plates on both sides
    plate = tr.Instance(Plate3D)
    
    # cavities on both sides
    cavity = tr.Instance(Cavities3D)
    
    # methode calculate the impedance matrix of the plate resonator 
    # accelerated by numba
    def zmatrix(self, height_d, M, freq):
        
        # cavity impedance matrix
        Zc = self.cavity.cavityimpedance(self.length, self.depth, self.j, self.k, self.l, self.n, freq)
        
        # calculate impedance matrix of plate induced sound
        # accelerated by numba
        Zprad = get_zprad_twosided(self.cavity.medium.c, self.cavity.medium.rho0, self.j, self.k, self.l, self.n, self.cavity.s, self.t, self.length, self.depth, height_d, M, freq)
        
        # calculate the total impedance matrix of the lining
        Z = Zc+Zprad
        
        return Z
    
    # method calculate the plate velocity
    def platevelocity(self, height_d, I, M, freq):
        
        # impedance matrix
        Z = self.zmatrix(height_d, M, freq)
        
        # L matrix with material properties of the plate
        Lmatrix = self.plate.lmatrix(self.length, self.depth, self.l, self.n, freq)
        
        # solving the linear system of equations
        # building lhs matrix from Z and Lmatrix
        lhs = Z+Lmatrix
        
        #calculate the plate velocity
        vp = np.zeros_like(I, dtype = complex)
        for i in range(len(freq)):
            
            vp[:,:,i] = np.linalg.tensorsolve(lhs[:,:,:,:,i], -I[:,:,i], axes=(0,1))
        
        return vp
    
    # method calculates the transmission factor of the plate silencer
    def transmissionfactor(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended arrays of modes
        J = self.j[:, np.newaxis, np.newaxis]
        K = self.k[np.newaxis, :, np.newaxis]
        
        # calculate transmission coefficient
        x0=(np.pi)**(2+0j)*(J)**(2+0j)
        x1=self.length*k0/(M + 1) + numpy.pi*J
        
        temp = (1/2)*self.length*J*((-1)**(K + 1+0j) - numpy.exp(1j*x1) + numpy.exp(1j*(numpy.pi*K + x1)) + 1)/(medium.c*height_d*K*(-(self.length)**(2+0j)*(k0)**(2+0j) + (M)**(2+0j)*x0 + 2*M*x0 + x0))
        
        # two-sided plate silencer = 2*temp
        tra_fac = np.exp(-1j*((k0*self.length)/(1+M)))*(np.sum(vp*(2*temp), axis=(0,1))+1)
        
        return tra_fac
    
    # method calculates the transmittance of the plate silencer
    def transmittance(self, vp, height_d, I, M, medium, freq):
        
        # call transmission factor
        tra_fac = self.transmissionfactor(vp, height_d, I, M, medium, freq)
        
        # calculate transmittance
        tra = np.abs(tra_fac)**2
        
        return tra
    
    # method calculates the reflection factor of the plate silencer
    def reflectionfactor(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis, np.newaxis]
        K = self.k[np.newaxis, :, np.newaxis]
        
        # calculate reflection coefficient
        x0=self.length*k0/(M - 1)
        x1=1j*x0
        x2=numpy.pi**2*J**2
        
        temp = (1/2)*self.length*J*((-1)**J + (-1)**(J + K + 1) + numpy.exp(-1j*(-numpy.pi*K + x0)) - numpy.exp(-x1))*numpy.exp(x1)/(medium.c*height_d*K*(self.length**2*k0**2 - M**2*x2 + 2*M*x2 - x2))
        
        # two-sided plate silencer = 2*temp
        ref_fac = np.sum(vp*(2*temp), axis=(0,1))
        
        return ref_fac
    
    # method calculates the reflectance of the plate silencer
    def reflectance(self, vp, height_d, I, M, medium, freq):
        
        # call reflection factor
        ref_fac = self.reflectionfactor(vp, height_d, I, M, medium, freq)
        
        # calculate reflectance with energetic correction due mean flow
        ref = np.abs(ref_fac)**2*((1-M)/(1+M))**2
        
        return ref
    
    # method calculates the dissipation of the plate silencer
    def dissipation(self, vp, height_d, I, M, medium, freq):
        
        # call transmittance
        tra = self.transmittance(vp, height_d, I, M, medium, freq)
        
        # call reflectance
        ref = self.reflectance(vp, height_d, I, M, medium, freq)
        
        # calculate dissipation
        dis = 1-tra-ref
        
        return dis
    
    # method calculate the transmission loss of the plate silencer
    def transmissionloss(self, vp, height_d, I, M, medium, freq):
        
        # transmission coefficient
        tra = self.transmittance(vp, height_d, I, M, medium, freq)
        
        # transmission loss
        TL = -10*np.log10(tra)
        
        return TL
    
