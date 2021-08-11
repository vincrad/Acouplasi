#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:58:19 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
from Fluid import Fluid
from Cavity import Cavities2D, Cavities3D
from Plate import Plate, Plate3D

import numba as nb

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
    
    # method calculates the transmission coefficient of the plate silencer
    def transmission(self, vp, height_d, I, M, medium, freq):
        
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
        
        tau = np.abs(np.sum(vp*temp, axis=0)+1)**2
        
        return tau
    
    # method calculates the reflaction coefficient of the plate silencer
    def reflection(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis]
        
        # calculate reflactance
        x0=1j*self.length*k0/(M - 1)
        x1=numpy.pi**2*J**2
        
        temp = (1/2)*numpy.pi*self.length*J*((-1)**J - numpy.exp(-x0))*numpy.exp(x0)/(self.cavity.medium.c*height_d*(self.length**2*k0**2 - M**2*x1 + 2*M*x1 - x1))
        
        beta_temp = np.abs(np.sum(vp*temp, axis=0))**2
        
        # energetic correction of reflection coefficient due to mean flow
        beta = beta_temp*((1-M)/(1+M))**2
        
        return beta
    
    # method to calculate the absorption coefficient of the plate silencer
    def absorption(self, vp, height_d, I, M, medium, freq):
        
        tau = self.transmission(vp, height_d, I, M, medium, freq)
        
        beta = self.reflection(vp, height_d, I, M, medium, freq)
        
        alpha = 1-tau-beta
        
        return alpha
    
    # method calculates the transmission loss of the plate silencer
    def transmissionloss(self, vp, height_d, I, M, medium, freq):
        
        # transmission coefficient
        tau = self.transmission(vp, height_d, I, M, medium, freq)        

        # transmission loss
        TL = -10*np.log10(tau)        

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
    
    # method to calculate the transmission coefficient of the plate silencer
    def transmission(self, vp, height_d, I, M, medium, freq):
        
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
        
        tau = np.abs(np.sum(vp*(2*temp), axis=0)+1)**2
        
        return tau
    
    # method calculates the reflaction coefficient of the plate silencer
    def reflection(self, vp, height_d, I, M, medium, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/medium.c
        
        # define extended array of modes
        J = self.j[:, np.newaxis]
        
        # calculate reflactance
        x0=1j*self.length*k0/(M - 1)
        x1=numpy.pi**2*J**2
        
        temp = (1/2)*numpy.pi*self.length*J*((-1)**J - numpy.exp(-x0))*numpy.exp(x0)/(self.cavity.medium.c*height_d*(self.length**2*k0**2 - M**2*x1 + 2*M*x1 - x1))
        
        beta_temp = np.abs(np.sum(vp*(2*temp), axis=0))**2
        
        # energetic correction of reflection coefficient due to mean flow
        beta = beta_temp*((1-M)/(1+M))**2
        
        return beta
    
    # method to calculate the absorption coefficient of the plate silencer
    def absorption(self, vp, height_d, I, M, medium, freq):
        
        tau = self.transmission(vp, height_d, I, M, medium, freq)
        
        beta = self.reflection(vp, height_d, I, M, medium, freq)
        
        alpha = 1-tau-beta
        
        return alpha
        
    # method calculates the transmission loss of the plate silencer
    def transmissionloss(self, vp, height_d, I, M, medium, freq):
        
        # transmission coefficient
        tau = self.transmission(vp, height_d, I, M, medium, freq)        

        # transmission loss
        TL = -10*np.log10(tau)        

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
                                        
                                        # j=l, k=s, n=2s
                                        if N==2*S:
                                            
                                            Sum += 0
                                            
                                        else:
                                            
                                            x0=(np.pi)**(2+0j)
                                            x1=Kstm*M
                                            x2=Kstm*length
                                            x3=1j*x2
                                            x4=2*S
                                            x5=N + x4
                                            x6=numpy.pi*L
                                            x7=numpy.exp(x3)
                                            x8=1j*x7
                                            x9=k0*x8
                                            x10=(Kstm)**(3+0j)*(length)**(4+0j)*x9
                                            x11=(np.pi)**(4+0j)*M*(L)**(4+0j)
                                            x12=x11*x7
                                            x13=N + S
                                            x14=N + 1
                                            x15=(-1)**(x14+0j)
                                            x16=(-1)**(x13+0j)
                                            x17=(-1)**(S + 1+0j)
                                            x18=(L)**(2+0j)*x0
                                            x19=(length)**(2+0j)*x18
                                            x20=Kstm*x19*x9
                                            x21=(Kstm)**(2+0j)*M*x19*x8
                                            x22=(-1)**(L+0j)
                                            x23=2*k0
                                            x24=2*x1
                                            x25=L + 1
                                            x26=(-1)**(N + x25+0j)
                                            x27=k0*x26
                                            x28=L + S
                                            x29=(-1)**(x28 + 1+0j)
                                            x30=(-1)**(N + x28+0j)
                                            x31=x1*x26
                                            x32=1j*numpy.pi
                                            x33=x28*x32
                                            x34=3*x33
                                            x35=numpy.exp(x34)
                                            x36=3*L
                                            x37=(-1)**(x36 + x5 + 1+0j)
                                            x38=numpy.exp(x3 + x33)
                                            x39=k0*x22
                                            x40=1j*x6
                                            x41=numpy.exp(x3 + x40)
                                            x42=(-1)**(x25+0j)
                                            x43=k0*x42
                                            x44=(-1)**(L + N+0j)*x41
                                            x45=numpy.exp(x3 + x34)
                                            x46=numpy.exp(x3 + 3*x40)
                                            x47=x1*x22
                                            x48=x1*x42
                                            x49=(-1)**(x14 + x28+0j)*x46
                                            x50=x30*numpy.exp(x3 + x32*(S + x36))
                                            
                                            Sum += -1/4*1j*length*depth*c*rho0*(2 - deltas)*(2 - deltat)*(k0 + x1)*((N)**(2+0j) - 2*(S)**(2+0j))*((-1)**(N+0j)*x20 + (-1)**(S+0j)*x20 + (-1)**(N + 1/2+0j)*x12 + (-1)**(S + 1/2+0j)*x12 + (-1)**(S + x14+0j)*x20 + (-1)**(x13 + 3/2+0j)*x12 + length*x18*(k0*x29 - k0*x35 + k0*x37 + k0*x44 + k0*x49 + k0*x50 + x1*x29 - x1*x35 + x1*x37 + x1*x44 + x1*x49 + x1*x50 + x22*x23 + x22*x24 + x23*x30 + x24*x30 + x27*x38 + x27 + x31*x38 + x31 + x38*x39 + x38*x47 + x39*x45 + x41*x43 + x41*x48 + x43*x46 + x45*x47 + x46*x48) + x10*x15 + x10*x16 + x10*x17 + x10 - x11*x8 + x15*x21 + x16*x21 + x17*x21 - x20 + x21)*numpy.exp(-x3)/(height_d*k0*N*S*x0*x5*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(N - x4)*(x2 - x6)**(2+0j)*(x2 + x6)**(2+0j))

                                    # j=l, n=s
                                    elif N==S:
                                        
                                        x0=(-1)**(S+0j)
                                        x1=(np.pi)**(2+0j)
                                        x2=Kstm*M
                                        x3=numpy.pi*L
                                        x4=Kstm*length
                                        x5=1j*k0
                                        x6=(Kstm)**(3+0j)*(length)**(4+0j)*x5
                                        x7=(np.pi)**(4+0j)*M*(L)**(4+0j)
                                        x8=2*k0
                                        x9=(L)**(2+0j)*x1
                                        x10=length*x9
                                        x11=x10*x8
                                        x12=M*x9
                                        x13=x12*x4
                                        x14=2*x13
                                        x15=K + S
                                        x16=K + 1
                                        x17=(-1)**(x16+0j)
                                        x18=(-1)**(x15+0j)
                                        x19=(-1)**(S + 1+0j)
                                        x20=(length)**(2+0j)
                                        x21=Kstm*x20*x5*x9
                                        x22=1j*(Kstm)**(2+0j)*x12*x20
                                        x23=(-1)**(K + 2*S+0j)
                                        x24=k0*x10
                                        x25=4*S
                                        x26=(-1)**(K + x25+0j)
                                        x27=(-1)**(L+0j)
                                        x28=L + S
                                        x29=(-1)**(x28+0j)
                                        x30=(-1)**(L + x16+0j)
                                        x31=(-1)**(L + x15+0j)
                                        x32=2*x2
                                        x33=(-1)**(3*L + x16 + x25+0j)
                                        x34=x0*numpy.exp(3*1j*numpy.pi*x28)
                                        
                                        Sum += (1/12)*1j*length*depth*c*rho0*x0*(2 - deltas)*(2 - deltat)*(k0 + x2)*((-1)**(K+0j)*x21 + (-1)**(K + 1/2+0j)*x7 + (-1)**(S + 1/2+0j)*x7 + (-1)**(S + x16+0j)*x21 + (-1)**(x15 + 3/2+0j)*x7 + x0*x11 + x0*x14 + x0*x21 + x10*(k0*x27 + k0*x30 + k0*x33 + k0*x34 + x2*x27 + x2*x30 + x2*x33 + x2*x34 - x29*x32 - x29*x8 + x31*x32 + x31*x8)*numpy.exp(-1j*x4) - x11*x18 - x11 + x13*x23 + x13*x26 - x14*x18 - x14 + x17*x22 + x17*x6 + x18*x22 + x18*x6 + x19*x22 + x19*x6 - x21 + x22 + x23*x24 + x24*x26 + x6 - 1j*x7)/(height_d*K*k0*S*x1*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(-x3 + x4)**(2+0j)*(x3 + x4)**(2+0j))


                                    # j=l, n=2r
                                    elif N==2*S:
                                        
                                        # j=l, n=2r, k=r
                                        # intercepted above
                                        if K==S:
                                            pass
                                        
                                        else:
                                            
                                            x0=(np.pi)**(2+0j)
                                            x1=K + S
                                            x2=K - S
                                            x3=numpy.pi*L
                                            x4=Kstp*length
                                            x5=(x3 + x4)**(2+0j)
                                            x6=(-x3 + x4)**(2+0j)
                                            x7=(np.pi)**(4+0j)*(L)**(4+0j)
                                            x8=(length)**(4+0j)
                                            x9=(length)**(2+0j)
                                            x10=(L)**(2+0j)
                                            x11=x0*x10
                                            x12=2*x11
                                            x13=(Kstm)**(4+0j)*x8 - (Kstm)**(2+0j)*x12*x9 + x7
                                            x14=numpy.exp(1j*x4)
                                            x15=Kstm*M
                                            x16=K + L
                                            x17=(-1)**(x16+0j)
                                            x18=k0*x17
                                            x19=3*K + 1
                                            x20=1j*length
                                            x21=(-1)**(S+0j)
                                            x22=Kstp*M
                                            x23=2*(-1)**(L+0j)
                                            x24=L + S
                                            x25=M*x7
                                            x26=1j*k0
                                            x27=(Kstp)**(3+0j)*x8
                                            x28=length*k0
                                            x29=4*x11
                                            x30=M*x4
                                            x31=(-1)**(K + 1/2+0j)
                                            x32=(-1)**(K+0j)
                                            x33=x12*x28
                                            x34=x12*x30
                                            x35=x11*x9
                                            x36=(Kstp)**(2+0j)*M
                                            x37=(-1)**(K + 3/2+0j)*x35
                                            
                                            Sum += (1/48)*depth*c*rho0*x20*(deltas - 2)*(deltat - 2)*(-8*(K)**(2+0j)*x13*(-k0 + x22)*(x21 - 1)*(length*x12*((-1)**(x24+0j)*k0 + (-1)**(x16 + 1+0j)*x22 + (-1)**(x24 + 1+0j)*x22 + k0*x23 + x18 - x22*x23) + x14*(Kstp*k0*x37 - Kstp*x26*x35 + k0*x27*x31 - x21*x33 + x21*x34 + x25*x31 + 1j*x25 + x26*x27 - x28*x29 + x29*x30 - x32*x33 + x32*x34 - 1j*x35*x36 + x36*x37))*numpy.exp(Kstm*x20) + 3*(np.pi)**(3+0j)*S*x1*x10*x14*x2*x20*x5*x6*(k0 + x15)*((-1)**(5*L + x19+0j)*k0 + (-1)**(7*L + x19+0j)*x15 + x15*x17 + x18))*numpy.exp(-x20*(Kstm + Kstp))/(height_d*K*k0*S*x0*x1*x13*x2*x5*x6*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j))
                                            
                                    else:
                                        
                                        x0=K + S
                                        x1=-S
                                        x2=N + S
                                        x3=numpy.pi*L
                                        x4=Kstp*length
                                        x5=-x3
                                        x6=Kstp*M
                                        x7=1j*M
                                        x8=(np.pi)**(4+0j)*(L)**(4+0j)
                                        x9=x7*x8
                                        x10=1j*k0
                                        x11=(length)**(4+0j)
                                        x12=(Kstp)**(3+0j)*x11
                                        x13=x10*x12
                                        x14=(np.pi)**(2+0j)
                                        x15=(L)**(2+0j)*x14
                                        x16=length*x15
                                        x17=2*x16
                                        x18=k0*x17
                                        x19=M*x15
                                        x20=2*x19*x4
                                        x21=K + N
                                        x22=(-1)**(x21 + 1/2+0j)
                                        x23=M*x8
                                        x24=k0*x12
                                        x25=(-1)**(x0 + 3/2+0j)
                                        x26=(length)**(2+0j)
                                        x27=Kstp*x15
                                        x28=x10*x26*x27
                                        x29=(-1)**(x21+0j)
                                        x30=(-1)**(x0+0j)
                                        x31=(-1)**(x2+0j)
                                        x32=(Kstp)**(2+0j)
                                        x33=x15*x26
                                        x34=x33*x7
                                        x35=k0*x27
                                        x36=(-1)**(x21 + 3/2+0j)
                                        x37=x26*x36
                                        x38=(-1)**(x0 + 1/2+0j)*x26
                                        x39=x19*x32
                                        x40=(-1)**(L+0j)
                                        x41=k0*x40
                                        x42=(-1)**(L + x21+0j)
                                        x43=k0*x42
                                        x44=L + 1
                                        x45=(-1)**(x44+0j)
                                        x46=(-1)**(x2 + x44+0j)
                                        x47=k0*x46
                                        x48=(-1)**(L + x2+0j)
                                        x49=(1/2)*length*(depth)**(2+0j)/(k0*x14)
                                        x50=2*S
                                        x51=N + x50
                                        x52=Kstm*length
                                        x53=Kstm*M
                                        x54=1j*x52
                                        x55=numpy.exp(x54)
                                        x56=x10*x55
                                        x57=(Kstm)**(3+0j)*x11*x56
                                        x58=x23*x55
                                        x59=(-1)**(K + 1+0j)
                                        x60=(-1)**(N + 1+0j)
                                        x61=Kstm*x33*x56
                                        x62=(Kstm)**(2+0j)*x34*x55
                                        x63=x40*x53
                                        x64=(-1)**(K + x44+0j)
                                        x65=(-1)**(N + x44+0j)
                                        x66=k0*x65
                                        x67=x53*x65
                                        x68=1j*numpy.pi
                                        x69=x68*(K + L)
                                        x70=3*x69
                                        x71=numpy.exp(x70)
                                        x72=3*L
                                        x73=x51 + x72
                                        x74=(-1)**(x73 + 1+0j)
                                        x75=3*K
                                        x76=(-1)**(x73 + x75+0j)
                                        x77=numpy.exp(x54 + x69)
                                        x78=1j*x3
                                        x79=numpy.exp(x54 + x78)
                                        x80=k0*x45
                                        x81=(-1)**(L + N+0j)*x79
                                        x82=numpy.exp(x54 + x70)
                                        x83=numpy.exp(x54 + 3*x78)
                                        x84=x45*x53
                                        x85=S + x72
                                        x86=x48*numpy.exp(x54 + x68*x85)
                                        x87=numpy.exp(x54 + x68*(x75 + x85))
                                        
                                        Sum += (1/2)*1j*c*rho0*(2 - deltas)*(2 - deltat)*(-K*N*x49*(k0 - x6)*((-1)**(x2 + 1/2+0j)*x26*x39 + (-1)**(x2 + 1+0j)*x13 + (-1)**(x2 + 3/2+0j)*x23 + x13 + x17*((-1)**(L + x0+0j)*x6 + (-1)**(x0 + x44+0j)*k0 + (-1)**(x21 + x44+0j)*x6 + x41 + x43 + x45*x6 + x47 + x48*x6)*numpy.exp(-1j*x4) - x18*x29 + x18*x30 + x18*x31 - x18 + x20*x29 - x20*x30 - x20*x31 + x20 + x22*x23 + x22*x24 + x23*x25 + x24*x25 + x28*x31 - x28 - x32*x34 + x35*x37 + x35*x38 + x37*x39 + x38*x39 + x9)/(x0*x2*(K + x1)*(N + x1)*(x3 + x4)**(2+0j)*(x4 + x5)**(2+0j)) - x49*(k0 + x53)*((N)**(2+0j) - 2*(S)**(2+0j))*((-1)**(K+0j)*x61 + (-1)**(N+0j)*x61 + (-1)**(K + 1/2+0j)*x58 + (-1)**(N + 1/2+0j)*x58 + (-1)**(x21 + 1+0j)*x61 + x16*(k0*x64 - k0*x71 + k0*x74 + k0*x76 + k0*x81 + k0*x86 + x41*x77 + x41*x82 + 2*x41 + x42*x53 + x43 + x46*x53*x87 + x47*x87 + x53*x64 - x53*x71 + x53*x74 + x53*x76 + x53*x81 + x53*x86 + x63*x77 + x63*x82 + 2*x63 + x66*x77 + x66 + x67*x77 + x67 + x79*x80 + x79*x84 + x80*x83 + x83*x84) + x29*x57 + x29*x62 + x36*x58 - x55*x9 + x57*x59 + x57*x60 + x57 + x59*x62 + x60*x62 - x61 + x62)*numpy.exp(-x54)/(K*N*x51*(N - x50)*(x3 + x52)**(2+0j)*(x5 + x52)**(2+0j)))/(depth*height_d*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j))

                                # k=s
                                elif K==S:
                                    
                                    # k=s, n=2s
                                    if N==2*S:
                                        
                                        Sum += 0
                                        
                                    # k=s, j=l
                                    # intercepted above
                                    elif J==L:
                                        pass
                                    
                                    else:
                                        
                                        x0=(np.pi)**(2+0j)
                                        x1=J + L
                                        x2=2*S
                                        x3=N + x2
                                        x4=numpy.pi*J
                                        x5=Kstm*length
                                        x6=numpy.pi*L
                                        x7=Kstm*M
                                        x8=(-1)**(J+0j)
                                        x9=(L)**(2+0j)
                                        x10=2*k0
                                        x11=x10*x9
                                        x12=x0*x11
                                        x13=(length)**(2+0j)
                                        x14=x13*x8
                                        x15=(Kstm)**(2+0j)*x10
                                        x16=2*(Kstm)**(3+0j)*M
                                        x17=(-1)**(L+0j)
                                        x18=x13*x15
                                        x19=x13*x16
                                        x20=2*x7
                                        x21=x20*x9
                                        x22=x0*x21
                                        x23=(-1)**(J + N+0j)
                                        x24=J + S
                                        x25=(-1)**(x24+0j)
                                        x26=(-1)**(L + N+0j)
                                        x27=(-1)**(L + S+0j)
                                        x28=N + S
                                        x29=(-1)**(J + x28+0j)
                                        x30=(-1)**(L + x28+0j)
                                        x31=1j*x5
                                        x32=(J)**(2+0j)
                                        x33=(-1)**(N+0j)
                                        x34=k0*x9
                                        x35=(-1)**(S+0j)
                                        x36=x7*x9
                                        x37=N + 1
                                        x38=(-1)**(x37+0j)
                                        x39=k0*x32
                                        x40=(-1)**(x28+0j)
                                        x41=(-1)**(S + 1+0j)
                                        x42=S + x37
                                        x43=(-1)**(x42+0j)
                                        x44=x32*x7
                                        x45=2*J
                                        x46=(-1)**(x28 + x45+0j)
                                        x47=(-1)**(x42 + x45+0j)
                                        x48=4*J + x3
                                        x49=(-1)**(x48+0j)
                                        x50=(-1)**(x48 + 1+0j)
                                        x51=1j*numpy.pi
                                        x52=x24*x51
                                        x53=3*x52
                                        x54=numpy.exp(x53)
                                        x55=x54*x8
                                        x56=(-1)**(J + 1+0j)*x54
                                        x57=1j*x4
                                        x58=numpy.exp(x31 + x57)
                                        x59=(-1)**(x1+0j)
                                        x60=x34*x59
                                        x61=numpy.exp(x31 + 3*x57)
                                        x62=(-1)**(x1 + 1+0j)
                                        x63=x39*x62
                                        x64=(-1)**(N + x1+0j)
                                        x65=x58*x64
                                        x66=x36*x59
                                        x67=numpy.exp(x31 + x52)
                                        x68=x39*x59
                                        x69=(-1)**(x1 + x37+0j)
                                        x70=x58*x69
                                        x71=x44*x62
                                        x72=numpy.exp(x31 + x53)
                                        x73=x34*x62
                                        x74=x64*x67
                                        x75=(-1)**(x1 + x28+0j)
                                        x76=x61*x75
                                        x77=x44*x59
                                        x78=x67*x69
                                        x79=(-1)**(x1 + x42+0j)
                                        x80=x61*x79
                                        x81=x36*x62
                                        x82=numpy.exp(x31 + x51*(3*J + S))
                                        x83=x75*x82
                                        x84=x79*x82
                                        
                                        Sum += -1/4*1j*depth*c*J*L*rho0*x14*(2 - deltas)*(2 - deltat)*(k0 + x7)*((N)**(2+0j) - 2*(S)**(2+0j))*(x0*(x10*x32 - x11 + x20*x32 - x21 + x33*x34 + x33*x36 + x34*x35 + x34*x43 + x34*x47 + x34*x49 + x34*x55 + x34*x70 + x34*x74 + x34*x76 + x34*x84 + x35*x36 + x36*x43 + x36*x47 + x36*x49 + x36*x55 + x36*x70 + x36*x74 + x36*x76 + x36*x84 + x38*x39 + x38*x44 + x39*x40 + x39*x41 + x39*x46 + x39*x50 + x39*x56 + x39*x65 + x39*x78 + x39*x80 + x39*x83 + x40*x44 + x41*x44 + x44*x46 + x44*x50 + x44*x56 + x44*x65 + x44*x78 + x44*x80 + x44*x83 + x58*x60 + x58*x63 + x58*x66 + x58*x71 + x60*x61 + x61*x63 + x61*x66 + x61*x71 + x67*x68 + x67*x73 + x67*x77 + x67*x81 + x68*x72 + x72*x73 + x72*x77 + x72*x81)*numpy.exp(-x31) - x12*x17 - x12*x23 - x12*x25 + x12*x26 + x12*x27 + x12*x29 - x12*x30 + x12*x8 - x14*x15 - x14*x16 + x17*x18 + x17*x19 - x17*x22 + x18*x23 + x18*x25 - x18*x26 - x18*x27 - x18*x29 + x18*x30 + x19*x23 + x19*x25 - x19*x26 - x19*x27 - x19*x29 + x19*x30 - x22*x23 - x22*x25 + x22*x26 + x22*x27 + x22*x29 - x22*x30 + x22*x8)/(height_d*k0*N*S*x0*x1*x3*(J - L)*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(N - x2)*(-x4 + x5)*(x4 + x5)*(x5 - x6)*(x5 + x6))

                                # n=s
                                elif N==S:
                                    
                                    # n=s, j=l
                                    # intercepted above
                                    if J==L:
                                        
                                        pass
                                    
                                    else:
                                        
                                        x0=(np.pi)**(2+0j)
                                        x1=(length)**(2+0j)
                                        x2=J + L
                                        x3=numpy.pi*J
                                        x4=Kstm*length
                                        x5=numpy.pi*L
                                        x6=(L)**(2+0j)
                                        x7=x0*x6
                                        x8=2*x7
                                        x9=2*(Kstm)**(2+0j)*x1
                                        x10=(-1)**(K+0j)
                                        x11=(-1)**(S+0j)
                                        x12=(-1)**(x2+0j)
                                        x13=(J)**(2+0j)
                                        x14=x0*x13
                                        x15=2*x14
                                        x16=K + S
                                        x17=(-1)**(x16+0j)
                                        x18=(-1)**(S + x2+0j)
                                        x19=(-1)**(K + x2+0j)
                                        x20=3*S + x2
                                        x21=x16 + x2
                                        x22=(-1)**(x21 + 1+0j)
                                        x23=x20 + 1
                                        x24=J + 1
                                        
                                        Sum += -1/12*1j*depth*c*J*L*rho0*x1*(2 - deltas)*(2 - deltat)*(Kstm*M + k0)**(2+0j)*((-1)**(x20+0j)*x14 + (-1)**(x21+0j)*x9 + (-1)**(x23+0j)*x7 + (-1)**(K + x20+0j)*x7 + (-1)**(K + x23+0j)*x14 + 2*x0*((-1)**(J+0j)*x13 + (-1)**(x24+0j)*x6 + (-1)**(J + K+0j)*x6 + (-1)**(J + S+0j)*x6 + (-1)**(J + x16+0j)*x13 + (-1)**(K + x24+0j)*x13 + (-1)**(S + x24+0j)*x13 + (-1)**(x16 + x24+0j)*x6)*numpy.exp(-1j*x4) - x10*x8 + x10*x9 - x11*x8 + x11*x9 - x12*x15 + x12*x9 + x14*x18 + x14*x22 + x15*x19 + x17*x8 - x17*x9 + x18*x7 - x18*x9 - x19*x9 + x22*x7 + x8 - x9)/(height_d*K*k0*S*x0*x2*(J - L)*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(-x3 + x4)*(x3 + x4)*(x4 - x5)*(x4 + x5))
                                        
                                # n=2s
                                elif N==2*S:
                                    
                                    # n=2s, j=l
                                    # intercepted above
                                    if J==L:
                                        
                                        pass
                                    
                                    # n=2s, k=s
                                    # intercepted above
                                    elif K==S:
                                        
                                        pass
                                    
                                    else:
                                        
                                        x0=(np.pi)**(2+0j)
                                        x1=(length)**(2+0j)
                                        x2=(-1)**(S+0j)
                                        x3=J + L
                                        x4=Kstp*length
                                        x5=1j*x4
                                        x6=numpy.pi*J
                                        x7=numpy.pi*L
                                        x8=(J)**(2+0j)
                                        x9=(L)**(2+0j)
                                        x10=(Kstp)**(2+0j)*x1
                                        
                                        Sum += -1/3*1j*depth*c*J*K*L*rho0*x1*(deltas - 2)*(deltat - 2)*(x2 - 1)*(Kstp*M - k0)**(2+0j)*((-1)**(L + 1+0j)*x0*(x8 - x9) + ((-1)**(x3+0j)*x10 + (-1)**(x3 + 1+0j)*x0*x9 + x0*x8 - x10)*numpy.exp(x5))*((-1)**(K+0j) + x2 + 2)*numpy.exp(-x5)/(height_d*k0*S*x0*x3*(J - L)*(K - S)*(K + S)*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(x4 - x6)*(x4 + x6)*(x4 - x7)*(x4 + x7))


                                # jk!=lm
                                else:
                                    
                                    x0=-S
                                    x1=numpy.pi*J
                                    x2=Kstp*length
                                    x3=numpy.pi*L
                                    x4=-x1
                                    x5=-x3
                                    x6=Kstp*M
                                    x7=K + N
                                    x8=(-1)**(x7+0j)
                                    x9=K + 1
                                    x10=N + 1
                                    x11=(np.pi)**(2+0j)
                                    x12=(J)**(2+0j)
                                    x13=x11*x12
                                    x14=(length)**(2+0j)
                                    x15=(Kstp)**(2+0j)*x14
                                    x16=J + L
                                    x17=(-1)**(x16+0j)
                                    x18=(-1)**(x16 + 1+0j)
                                    x19=(L)**(2+0j)
                                    x20=x11*x19
                                    x21=L + 1
                                    x22=(depth)**(2+0j)*J*L*x14/(k0*x11*x16*(J - L))
                                    x23=(-1)**(J+0j)
                                    x24=2*S
                                    x25=N + x24
                                    x26=Kstm*length
                                    x27=2*x19
                                    x28=x11*x27
                                    x29=2*(Kstm)**(2+0j)*x14
                                    x30=(-1)**(L+0j)
                                    x31=J + K
                                    x32=(-1)**(x31+0j)
                                    x33=(-1)**(J + N+0j)
                                    x34=K + L
                                    x35=(-1)**(x34+0j)
                                    x36=L + N
                                    x37=(-1)**(x36+0j)
                                    x38=(-1)**(J + x7+0j)
                                    x39=(-1)**(L + x7+0j)
                                    x40=2*J
                                    x41=L + x40
                                    x42=(-1)**(x41+0j)
                                    x43=2*x12
                                    x44=x36 + x40
                                    x45=4*J + x25
                                    x46=L + x45
                                    x47=numpy.exp(3*1j*numpy.pi*x31)
                                    x48=3*K
                                    x49=x45 + x48
                                    x50=x45 + 1
                                    
                                    Sum += (1/2)*1j*c*rho0*(2 - deltas)*(2 - deltat)*(-K*N*x22*(-k0 + x6)*(k0 - x6)*((-1)**(S + x10+0j) + (-1)**(S + x9+0j) + x8 + 1)*((-1)**(x21+0j)*x11*(x12 - x19)*numpy.exp(-1j*x2) + x13 + x15*x17 - x15 + x18*x20)/((K + S)*(K + x0)*(N + S)*(N + x0)*(x1 + x2)*(x2 + x3)*(x2 + x4)*(x2 + x5)) + (1/2)*x22*x23*((N)**(2+0j) - 2*(S)**(2+0j))*(Kstm*M + k0)**(2+0j)*((-1)**(x44+0j)*x20 + (-1)**(x46+0j)*x20 + (-1)**(x10 + x41+0j)*x13 + (-1)**(x21 + x45+0j)*x13 + (-1)**(x21 + x49+0j)*x20 + (-1)**(x34 + x40+0j)*x20 + (-1)**(x41 + x7+0j)*x13 + (-1)**(x41 + x9+0j)*x13 + (-1)**(x44 + x9+0j)*x20 + (-1)**(x46 + x48+0j)*x13 + x11*x42*x43 + x11*((-1)**(K+0j)*x12 + (-1)**(N+0j)*x12 + (-1)**(x10+0j)*x19 + (-1)**(x45+0j)*x12 + (-1)**(x49+0j)*x19 + (-1)**(x50+0j)*x19 + (-1)**(x9+0j)*x19 + (-1)**(J + 1+0j)*x19*x47 + (-1)**(N + x9+0j)*x12 + (-1)**(x48 + x50+0j)*x12 + x12*x23*x47 + x19*x8 + x27 - x43)*numpy.exp(-1j*x26) + x13*x18*x47 + x17*x20*x47 - x23*x28 + x23*x29 + x28*x30 + x28*x32 + x28*x33 - x28*x35 - x28*x37 - x28*x38 + x28*x39 - x28*x42 - x29*x30 - x29*x32 - x29*x33 + x29*x35 + x29*x37 + x29*x38 - x29*x39)/(K*N*x25*(N - x24)*(x1 + x26)*(x26 + x3)*(x26 + x4)*(x26 + x5)))/(depth*height_d*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j))
                                
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
        
    # method to calculate the transmission coefficient of the plate silencer
    def transmission(self, vp, height_d, I, M, medium, freq):
        
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
        
        tau = np.abs(np.sum(vp*temp, axis=(0,1))+1)**2
        
        return tau
    
    # method to calculate the reflection coefficient of the plate silencer
    def reflection(self, vp, height_d, I, M, medium, freq):
        
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
        
        beta_temp = np.abs(np.sum(vp*temp, axis=(0,1)))**2
        
        # energetic correction of reflection coefficient due to mean flow
        beta = beta_temp*((1-M)/(1+M))**2
        
        return beta
    
    # method to calculate the absorption coefficient of the plate silencer
    def absorption(self, vp, height_d, I, M, medium, freq):
        
        # transmission coefficient
        tau = self.transmission(vp, height_d, I, M, medium, freq)
        
        # reflection coefficient
        beta = self.reflection(vp, height_d, I, M, medium, freq)
        
        # absorption coefficient
        alpha = 1-tau-beta
        
        return alpha
    
    # method calculate the transmission loss of the plate silencer
    def transmissionloss(self, vp, height_d, I, M, medium, freq):
        
        # transmission coefficient
        tau = self.transmission(vp, height_d, I, M, medium, freq)
        
        # transmission loss
        TL = -10*np.log10(tau)
        
        return TL

#%% SimpleTwoSidedPlateResonator3D

# method to calculate impedance matrix of plate induced sound for SimpleTwoSidedPlateResonator3D
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
                                        
                                        # j=l, k=s, n=2s
                                        if N==2*S:
                                            
                                            Sum += 0
                                            
                                        else:
                                            
                                            x0=(np.pi)**(2+0j)
                                            x1=Kstm*M
                                            x2=Kstm*length
                                            x3=1j*x2
                                            x4=2*S
                                            x5=N + x4
                                            x6=numpy.pi*L
                                            x7=numpy.exp(x3)
                                            x8=1j*x7
                                            x9=k0*x8
                                            x10=(Kstm)**(3+0j)*(length)**(4+0j)*x9
                                            x11=(np.pi)**(4+0j)*M*(L)**(4+0j)
                                            x12=x11*x7
                                            x13=N + S
                                            x14=N + 1
                                            x15=(-1)**(x14+0j)
                                            x16=(-1)**(x13+0j)
                                            x17=(-1)**(S + 1+0j)
                                            x18=(L)**(2+0j)*x0
                                            x19=(length)**(2+0j)*x18
                                            x20=Kstm*x19*x9
                                            x21=(Kstm)**(2+0j)*M*x19*x8
                                            x22=(-1)**(L+0j)
                                            x23=2*k0
                                            x24=2*x1
                                            x25=L + 1
                                            x26=(-1)**(N + x25+0j)
                                            x27=k0*x26
                                            x28=L + S
                                            x29=(-1)**(x28 + 1+0j)
                                            x30=(-1)**(N + x28+0j)
                                            x31=x1*x26
                                            x32=1j*numpy.pi
                                            x33=x28*x32
                                            x34=3*x33
                                            x35=numpy.exp(x34)
                                            x36=3*L
                                            x37=(-1)**(x36 + x5 + 1+0j)
                                            x38=numpy.exp(x3 + x33)
                                            x39=k0*x22
                                            x40=1j*x6
                                            x41=numpy.exp(x3 + x40)
                                            x42=(-1)**(x25+0j)
                                            x43=k0*x42
                                            x44=(-1)**(L + N+0j)*x41
                                            x45=numpy.exp(x3 + x34)
                                            x46=numpy.exp(x3 + 3*x40)
                                            x47=x1*x22
                                            x48=x1*x42
                                            x49=(-1)**(x14 + x28+0j)*x46
                                            x50=x30*numpy.exp(x3 + x32*(S + x36))
                                            
                                            # calculation with additional factor (1+(-1)**T) for two sided plate silencer
                                            Sum += (1+(-1)**T)*(-1/4*1j*length*depth*c*rho0*(2 - deltas)*(2 - deltat)*(k0 + x1)*((N)**(2+0j) - 2*(S)**(2+0j))*((-1)**(N+0j)*x20 + (-1)**(S+0j)*x20 + (-1)**(N + 1/2+0j)*x12 + (-1)**(S + 1/2+0j)*x12 + (-1)**(S + x14+0j)*x20 + (-1)**(x13 + 3/2+0j)*x12 + length*x18*(k0*x29 - k0*x35 + k0*x37 + k0*x44 + k0*x49 + k0*x50 + x1*x29 - x1*x35 + x1*x37 + x1*x44 + x1*x49 + x1*x50 + x22*x23 + x22*x24 + x23*x30 + x24*x30 + x27*x38 + x27 + x31*x38 + x31 + x38*x39 + x38*x47 + x39*x45 + x41*x43 + x41*x48 + x43*x46 + x45*x47 + x46*x48) + x10*x15 + x10*x16 + x10*x17 + x10 - x11*x8 + x15*x21 + x16*x21 + x17*x21 - x20 + x21)*numpy.exp(-x3)/(height_d*k0*N*S*x0*x5*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(N - x4)*(x2 - x6)**(2+0j)*(x2 + x6)**(2+0j)))

                                    # j=l, n=s
                                    elif N==S:
                                        
                                        x0=(-1)**(S+0j)
                                        x1=(np.pi)**(2+0j)
                                        x2=Kstm*M
                                        x3=numpy.pi*L
                                        x4=Kstm*length
                                        x5=1j*k0
                                        x6=(Kstm)**(3+0j)*(length)**(4+0j)*x5
                                        x7=(np.pi)**(4+0j)*M*(L)**(4+0j)
                                        x8=2*k0
                                        x9=(L)**(2+0j)*x1
                                        x10=length*x9
                                        x11=x10*x8
                                        x12=M*x9
                                        x13=x12*x4
                                        x14=2*x13
                                        x15=K + S
                                        x16=K + 1
                                        x17=(-1)**(x16+0j)
                                        x18=(-1)**(x15+0j)
                                        x19=(-1)**(S + 1+0j)
                                        x20=(length)**(2+0j)
                                        x21=Kstm*x20*x5*x9
                                        x22=1j*(Kstm)**(2+0j)*x12*x20
                                        x23=(-1)**(K + 2*S+0j)
                                        x24=k0*x10
                                        x25=4*S
                                        x26=(-1)**(K + x25+0j)
                                        x27=(-1)**(L+0j)
                                        x28=L + S
                                        x29=(-1)**(x28+0j)
                                        x30=(-1)**(L + x16+0j)
                                        x31=(-1)**(L + x15+0j)
                                        x32=2*x2
                                        x33=(-1)**(3*L + x16 + x25+0j)
                                        x34=x0*numpy.exp(3*1j*numpy.pi*x28)
                                        
                                        # calculation with additional factor (1+(-1)**T) for two sided plate silencer
                                        Sum += (1+(-1)**T)*((1/12)*1j*length*depth*c*rho0*x0*(2 - deltas)*(2 - deltat)*(k0 + x2)*((-1)**(K+0j)*x21 + (-1)**(K + 1/2+0j)*x7 + (-1)**(S + 1/2+0j)*x7 + (-1)**(S + x16+0j)*x21 + (-1)**(x15 + 3/2+0j)*x7 + x0*x11 + x0*x14 + x0*x21 + x10*(k0*x27 + k0*x30 + k0*x33 + k0*x34 + x2*x27 + x2*x30 + x2*x33 + x2*x34 - x29*x32 - x29*x8 + x31*x32 + x31*x8)*numpy.exp(-1j*x4) - x11*x18 - x11 + x13*x23 + x13*x26 - x14*x18 - x14 + x17*x22 + x17*x6 + x18*x22 + x18*x6 + x19*x22 + x19*x6 - x21 + x22 + x23*x24 + x24*x26 + x6 - 1j*x7)/(height_d*K*k0*S*x1*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(-x3 + x4)**(2+0j)*(x3 + x4)**(2+0j)))


                                    # j=l, n=2r
                                    elif N==2*S:
                                        
                                        # j=l, n=2r, k=r
                                        # intercepted above
                                        if K==S:
                                            pass
                                        
                                        else:
                                            
                                            x0=(np.pi)**(2+0j)
                                            x1=K + S
                                            x2=K - S
                                            x3=numpy.pi*L
                                            x4=Kstp*length
                                            x5=(x3 + x4)**(2+0j)
                                            x6=(-x3 + x4)**(2+0j)
                                            x7=(np.pi)**(4+0j)*(L)**(4+0j)
                                            x8=(length)**(4+0j)
                                            x9=(length)**(2+0j)
                                            x10=(L)**(2+0j)
                                            x11=x0*x10
                                            x12=2*x11
                                            x13=(Kstm)**(4+0j)*x8 - (Kstm)**(2+0j)*x12*x9 + x7
                                            x14=numpy.exp(1j*x4)
                                            x15=Kstm*M
                                            x16=K + L
                                            x17=(-1)**(x16+0j)
                                            x18=k0*x17
                                            x19=3*K + 1
                                            x20=1j*length
                                            x21=(-1)**(S+0j)
                                            x22=Kstp*M
                                            x23=2*(-1)**(L+0j)
                                            x24=L + S
                                            x25=M*x7
                                            x26=1j*k0
                                            x27=(Kstp)**(3+0j)*x8
                                            x28=length*k0
                                            x29=4*x11
                                            x30=M*x4
                                            x31=(-1)**(K + 1/2+0j)
                                            x32=(-1)**(K+0j)
                                            x33=x12*x28
                                            x34=x12*x30
                                            x35=x11*x9
                                            x36=(Kstp)**(2+0j)*M
                                            x37=(-1)**(K + 3/2+0j)*x35
                                            
                                            # calculation with additional factor (1+(-1)**T) for two sided plate silencer
                                            Sum += (1+(-1)**T)*((1/48)*depth*c*rho0*x20*(deltas - 2)*(deltat - 2)*(-8*(K)**(2+0j)*x13*(-k0 + x22)*(x21 - 1)*(length*x12*((-1)**(x24+0j)*k0 + (-1)**(x16 + 1+0j)*x22 + (-1)**(x24 + 1+0j)*x22 + k0*x23 + x18 - x22*x23) + x14*(Kstp*k0*x37 - Kstp*x26*x35 + k0*x27*x31 - x21*x33 + x21*x34 + x25*x31 + 1j*x25 + x26*x27 - x28*x29 + x29*x30 - x32*x33 + x32*x34 - 1j*x35*x36 + x36*x37))*numpy.exp(Kstm*x20) + 3*(np.pi)**(3+0j)*S*x1*x10*x14*x2*x20*x5*x6*(k0 + x15)*((-1)**(5*L + x19+0j)*k0 + (-1)**(7*L + x19+0j)*x15 + x15*x17 + x18))*numpy.exp(-x20*(Kstm + Kstp))/(height_d*K*k0*S*x0*x1*x13*x2*x5*x6*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)))
                                            
                                    else:
                                        
                                        x0=K + S
                                        x1=-S
                                        x2=N + S
                                        x3=numpy.pi*L
                                        x4=Kstp*length
                                        x5=-x3
                                        x6=Kstp*M
                                        x7=1j*M
                                        x8=(np.pi)**(4+0j)*(L)**(4+0j)
                                        x9=x7*x8
                                        x10=1j*k0
                                        x11=(length)**(4+0j)
                                        x12=(Kstp)**(3+0j)*x11
                                        x13=x10*x12
                                        x14=(np.pi)**(2+0j)
                                        x15=(L)**(2+0j)*x14
                                        x16=length*x15
                                        x17=2*x16
                                        x18=k0*x17
                                        x19=M*x15
                                        x20=2*x19*x4
                                        x21=K + N
                                        x22=(-1)**(x21 + 1/2+0j)
                                        x23=M*x8
                                        x24=k0*x12
                                        x25=(-1)**(x0 + 3/2+0j)
                                        x26=(length)**(2+0j)
                                        x27=Kstp*x15
                                        x28=x10*x26*x27
                                        x29=(-1)**(x21+0j)
                                        x30=(-1)**(x0+0j)
                                        x31=(-1)**(x2+0j)
                                        x32=(Kstp)**(2+0j)
                                        x33=x15*x26
                                        x34=x33*x7
                                        x35=k0*x27
                                        x36=(-1)**(x21 + 3/2+0j)
                                        x37=x26*x36
                                        x38=(-1)**(x0 + 1/2+0j)*x26
                                        x39=x19*x32
                                        x40=(-1)**(L+0j)
                                        x41=k0*x40
                                        x42=(-1)**(L + x21+0j)
                                        x43=k0*x42
                                        x44=L + 1
                                        x45=(-1)**(x44+0j)
                                        x46=(-1)**(x2 + x44+0j)
                                        x47=k0*x46
                                        x48=(-1)**(L + x2+0j)
                                        x49=(1/2)*length*(depth)**(2+0j)/(k0*x14)
                                        x50=2*S
                                        x51=N + x50
                                        x52=Kstm*length
                                        x53=Kstm*M
                                        x54=1j*x52
                                        x55=numpy.exp(x54)
                                        x56=x10*x55
                                        x57=(Kstm)**(3+0j)*x11*x56
                                        x58=x23*x55
                                        x59=(-1)**(K + 1+0j)
                                        x60=(-1)**(N + 1+0j)
                                        x61=Kstm*x33*x56
                                        x62=(Kstm)**(2+0j)*x34*x55
                                        x63=x40*x53
                                        x64=(-1)**(K + x44+0j)
                                        x65=(-1)**(N + x44+0j)
                                        x66=k0*x65
                                        x67=x53*x65
                                        x68=1j*numpy.pi
                                        x69=x68*(K + L)
                                        x70=3*x69
                                        x71=numpy.exp(x70)
                                        x72=3*L
                                        x73=x51 + x72
                                        x74=(-1)**(x73 + 1+0j)
                                        x75=3*K
                                        x76=(-1)**(x73 + x75+0j)
                                        x77=numpy.exp(x54 + x69)
                                        x78=1j*x3
                                        x79=numpy.exp(x54 + x78)
                                        x80=k0*x45
                                        x81=(-1)**(L + N+0j)*x79
                                        x82=numpy.exp(x54 + x70)
                                        x83=numpy.exp(x54 + 3*x78)
                                        x84=x45*x53
                                        x85=S + x72
                                        x86=x48*numpy.exp(x54 + x68*x85)
                                        x87=numpy.exp(x54 + x68*(x75 + x85))
                                        
                                        # calculation with additional factor (1+(-1)**T) for two sided plate silencer
                                        Sum += (1+(-1)**T)*((1/2)*1j*c*rho0*(2 - deltas)*(2 - deltat)*(-K*N*x49*(k0 - x6)*((-1)**(x2 + 1/2+0j)*x26*x39 + (-1)**(x2 + 1+0j)*x13 + (-1)**(x2 + 3/2+0j)*x23 + x13 + x17*((-1)**(L + x0+0j)*x6 + (-1)**(x0 + x44+0j)*k0 + (-1)**(x21 + x44+0j)*x6 + x41 + x43 + x45*x6 + x47 + x48*x6)*numpy.exp(-1j*x4) - x18*x29 + x18*x30 + x18*x31 - x18 + x20*x29 - x20*x30 - x20*x31 + x20 + x22*x23 + x22*x24 + x23*x25 + x24*x25 + x28*x31 - x28 - x32*x34 + x35*x37 + x35*x38 + x37*x39 + x38*x39 + x9)/(x0*x2*(K + x1)*(N + x1)*(x3 + x4)**(2+0j)*(x4 + x5)**(2+0j)) - x49*(k0 + x53)*((N)**(2+0j) - 2*(S)**(2+0j))*((-1)**(K+0j)*x61 + (-1)**(N+0j)*x61 + (-1)**(K + 1/2+0j)*x58 + (-1)**(N + 1/2+0j)*x58 + (-1)**(x21 + 1+0j)*x61 + x16*(k0*x64 - k0*x71 + k0*x74 + k0*x76 + k0*x81 + k0*x86 + x41*x77 + x41*x82 + 2*x41 + x42*x53 + x43 + x46*x53*x87 + x47*x87 + x53*x64 - x53*x71 + x53*x74 + x53*x76 + x53*x81 + x53*x86 + x63*x77 + x63*x82 + 2*x63 + x66*x77 + x66 + x67*x77 + x67 + x79*x80 + x79*x84 + x80*x83 + x83*x84) + x29*x57 + x29*x62 + x36*x58 - x55*x9 + x57*x59 + x57*x60 + x57 + x59*x62 + x60*x62 - x61 + x62)*numpy.exp(-x54)/(K*N*x51*(N - x50)*(x3 + x52)**(2+0j)*(x5 + x52)**(2+0j)))/(depth*height_d*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)))

                                # k=s
                                elif K==S:
                                    
                                    # k=s, n=2s
                                    if N==2*S:
                                        
                                        Sum += 0
                                        
                                    # k=s, j=l
                                    # intercepted above
                                    elif J==L:
                                        pass
                                    
                                    else:
                                        
                                        x0=(np.pi)**(2+0j)
                                        x1=J + L
                                        x2=2*S
                                        x3=N + x2
                                        x4=numpy.pi*J
                                        x5=Kstm*length
                                        x6=numpy.pi*L
                                        x7=Kstm*M
                                        x8=(-1)**(J+0j)
                                        x9=(L)**(2+0j)
                                        x10=2*k0
                                        x11=x10*x9
                                        x12=x0*x11
                                        x13=(length)**(2+0j)
                                        x14=x13*x8
                                        x15=(Kstm)**(2+0j)*x10
                                        x16=2*(Kstm)**(3+0j)*M
                                        x17=(-1)**(L+0j)
                                        x18=x13*x15
                                        x19=x13*x16
                                        x20=2*x7
                                        x21=x20*x9
                                        x22=x0*x21
                                        x23=(-1)**(J + N+0j)
                                        x24=J + S
                                        x25=(-1)**(x24+0j)
                                        x26=(-1)**(L + N+0j)
                                        x27=(-1)**(L + S+0j)
                                        x28=N + S
                                        x29=(-1)**(J + x28+0j)
                                        x30=(-1)**(L + x28+0j)
                                        x31=1j*x5
                                        x32=(J)**(2+0j)
                                        x33=(-1)**(N+0j)
                                        x34=k0*x9
                                        x35=(-1)**(S+0j)
                                        x36=x7*x9
                                        x37=N + 1
                                        x38=(-1)**(x37+0j)
                                        x39=k0*x32
                                        x40=(-1)**(x28+0j)
                                        x41=(-1)**(S + 1+0j)
                                        x42=S + x37
                                        x43=(-1)**(x42+0j)
                                        x44=x32*x7
                                        x45=2*J
                                        x46=(-1)**(x28 + x45+0j)
                                        x47=(-1)**(x42 + x45+0j)
                                        x48=4*J + x3
                                        x49=(-1)**(x48+0j)
                                        x50=(-1)**(x48 + 1+0j)
                                        x51=1j*numpy.pi
                                        x52=x24*x51
                                        x53=3*x52
                                        x54=numpy.exp(x53)
                                        x55=x54*x8
                                        x56=(-1)**(J + 1+0j)*x54
                                        x57=1j*x4
                                        x58=numpy.exp(x31 + x57)
                                        x59=(-1)**(x1+0j)
                                        x60=x34*x59
                                        x61=numpy.exp(x31 + 3*x57)
                                        x62=(-1)**(x1 + 1+0j)
                                        x63=x39*x62
                                        x64=(-1)**(N + x1+0j)
                                        x65=x58*x64
                                        x66=x36*x59
                                        x67=numpy.exp(x31 + x52)
                                        x68=x39*x59
                                        x69=(-1)**(x1 + x37+0j)
                                        x70=x58*x69
                                        x71=x44*x62
                                        x72=numpy.exp(x31 + x53)
                                        x73=x34*x62
                                        x74=x64*x67
                                        x75=(-1)**(x1 + x28+0j)
                                        x76=x61*x75
                                        x77=x44*x59
                                        x78=x67*x69
                                        x79=(-1)**(x1 + x42+0j)
                                        x80=x61*x79
                                        x81=x36*x62
                                        x82=numpy.exp(x31 + x51*(3*J + S))
                                        x83=x75*x82
                                        x84=x79*x82
                                        
                                        # calculation with additional factor (1+(-1)**T) for two sided plate silencer
                                        Sum += (1+(-1)**T)*(-1/4*1j*depth*c*J*L*rho0*x14*(2 - deltas)*(2 - deltat)*(k0 + x7)*((N)**(2+0j) - 2*(S)**(2+0j))*(x0*(x10*x32 - x11 + x20*x32 - x21 + x33*x34 + x33*x36 + x34*x35 + x34*x43 + x34*x47 + x34*x49 + x34*x55 + x34*x70 + x34*x74 + x34*x76 + x34*x84 + x35*x36 + x36*x43 + x36*x47 + x36*x49 + x36*x55 + x36*x70 + x36*x74 + x36*x76 + x36*x84 + x38*x39 + x38*x44 + x39*x40 + x39*x41 + x39*x46 + x39*x50 + x39*x56 + x39*x65 + x39*x78 + x39*x80 + x39*x83 + x40*x44 + x41*x44 + x44*x46 + x44*x50 + x44*x56 + x44*x65 + x44*x78 + x44*x80 + x44*x83 + x58*x60 + x58*x63 + x58*x66 + x58*x71 + x60*x61 + x61*x63 + x61*x66 + x61*x71 + x67*x68 + x67*x73 + x67*x77 + x67*x81 + x68*x72 + x72*x73 + x72*x77 + x72*x81)*numpy.exp(-x31) - x12*x17 - x12*x23 - x12*x25 + x12*x26 + x12*x27 + x12*x29 - x12*x30 + x12*x8 - x14*x15 - x14*x16 + x17*x18 + x17*x19 - x17*x22 + x18*x23 + x18*x25 - x18*x26 - x18*x27 - x18*x29 + x18*x30 + x19*x23 + x19*x25 - x19*x26 - x19*x27 - x19*x29 + x19*x30 - x22*x23 - x22*x25 + x22*x26 + x22*x27 + x22*x29 - x22*x30 + x22*x8)/(height_d*k0*N*S*x0*x1*x3*(J - L)*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(N - x2)*(-x4 + x5)*(x4 + x5)*(x5 - x6)*(x5 + x6)))

                                # n=s
                                elif N==S:
                                    
                                    # n=s, j=l
                                    # intercepted above
                                    if J==L:
                                        
                                        pass
                                    
                                    else:
                                        
                                        x0=(np.pi)**(2+0j)
                                        x1=(length)**(2+0j)
                                        x2=J + L
                                        x3=numpy.pi*J
                                        x4=Kstm*length
                                        x5=numpy.pi*L
                                        x6=(L)**(2+0j)
                                        x7=x0*x6
                                        x8=2*x7
                                        x9=2*(Kstm)**(2+0j)*x1
                                        x10=(-1)**(K+0j)
                                        x11=(-1)**(S+0j)
                                        x12=(-1)**(x2+0j)
                                        x13=(J)**(2+0j)
                                        x14=x0*x13
                                        x15=2*x14
                                        x16=K + S
                                        x17=(-1)**(x16+0j)
                                        x18=(-1)**(S + x2+0j)
                                        x19=(-1)**(K + x2+0j)
                                        x20=3*S + x2
                                        x21=x16 + x2
                                        x22=(-1)**(x21 + 1+0j)
                                        x23=x20 + 1
                                        x24=J + 1
                                        
                                        # calculation with additional factor (1+(-1)**T) for two sided plate silencer
                                        Sum += (1+(-1)**T)*(-1/12*1j*depth*c*J*L*rho0*x1*(2 - deltas)*(2 - deltat)*(Kstm*M + k0)**(2+0j)*((-1)**(x20+0j)*x14 + (-1)**(x21+0j)*x9 + (-1)**(x23+0j)*x7 + (-1)**(K + x20+0j)*x7 + (-1)**(K + x23+0j)*x14 + 2*x0*((-1)**(J+0j)*x13 + (-1)**(x24+0j)*x6 + (-1)**(J + K+0j)*x6 + (-1)**(J + S+0j)*x6 + (-1)**(J + x16+0j)*x13 + (-1)**(K + x24+0j)*x13 + (-1)**(S + x24+0j)*x13 + (-1)**(x16 + x24+0j)*x6)*numpy.exp(-1j*x4) - x10*x8 + x10*x9 - x11*x8 + x11*x9 - x12*x15 + x12*x9 + x14*x18 + x14*x22 + x15*x19 + x17*x8 - x17*x9 + x18*x7 - x18*x9 - x19*x9 + x22*x7 + x8 - x9)/(height_d*K*k0*S*x0*x2*(J - L)*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(-x3 + x4)*(x3 + x4)*(x4 - x5)*(x4 + x5)))
                                        
                                # n=2s
                                elif N==2*S:
                                    
                                    # n=2s, j=l
                                    # intercepted above
                                    if J==L:
                                        
                                        pass
                                    
                                    # n=2s, k=s
                                    # intercepted above
                                    elif K==S:
                                        
                                        pass
                                    
                                    else:
                                        
                                        x0=(np.pi)**(2+0j)
                                        x1=(length)**(2+0j)
                                        x2=(-1)**(S+0j)
                                        x3=J + L
                                        x4=Kstp*length
                                        x5=1j*x4
                                        x6=numpy.pi*J
                                        x7=numpy.pi*L
                                        x8=(J)**(2+0j)
                                        x9=(L)**(2+0j)
                                        x10=(Kstp)**(2+0j)*x1
                                        
                                        # calculation with additional factor (1+(-1)**T) for two sided plate silencer
                                        Sum += (1+(-1)**T)*(-1/3*1j*depth*c*J*K*L*rho0*x1*(deltas - 2)*(deltat - 2)*(x2 - 1)*(Kstp*M - k0)**(2+0j)*((-1)**(L + 1+0j)*x0*(x8 - x9) + ((-1)**(x3+0j)*x10 + (-1)**(x3 + 1+0j)*x0*x9 + x0*x8 - x10)*numpy.exp(x5))*((-1)**(K+0j) + x2 + 2)*numpy.exp(-x5)/(height_d*k0*S*x0*x3*(J - L)*(K - S)*(K + S)*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)*(x4 - x6)*(x4 + x6)*(x4 - x7)*(x4 + x7)))


                                # jk!=lm
                                else:
                                    
                                    x0=-S
                                    x1=numpy.pi*J
                                    x2=Kstp*length
                                    x3=numpy.pi*L
                                    x4=-x1
                                    x5=-x3
                                    x6=Kstp*M
                                    x7=K + N
                                    x8=(-1)**(x7+0j)
                                    x9=K + 1
                                    x10=N + 1
                                    x11=(np.pi)**(2+0j)
                                    x12=(J)**(2+0j)
                                    x13=x11*x12
                                    x14=(length)**(2+0j)
                                    x15=(Kstp)**(2+0j)*x14
                                    x16=J + L
                                    x17=(-1)**(x16+0j)
                                    x18=(-1)**(x16 + 1+0j)
                                    x19=(L)**(2+0j)
                                    x20=x11*x19
                                    x21=L + 1
                                    x22=(depth)**(2+0j)*J*L*x14/(k0*x11*x16*(J - L))
                                    x23=(-1)**(J+0j)
                                    x24=2*S
                                    x25=N + x24
                                    x26=Kstm*length
                                    x27=2*x19
                                    x28=x11*x27
                                    x29=2*(Kstm)**(2+0j)*x14
                                    x30=(-1)**(L+0j)
                                    x31=J + K
                                    x32=(-1)**(x31+0j)
                                    x33=(-1)**(J + N+0j)
                                    x34=K + L
                                    x35=(-1)**(x34+0j)
                                    x36=L + N
                                    x37=(-1)**(x36+0j)
                                    x38=(-1)**(J + x7+0j)
                                    x39=(-1)**(L + x7+0j)
                                    x40=2*J
                                    x41=L + x40
                                    x42=(-1)**(x41+0j)
                                    x43=2*x12
                                    x44=x36 + x40
                                    x45=4*J + x25
                                    x46=L + x45
                                    x47=numpy.exp(3*1j*numpy.pi*x31)
                                    x48=3*K
                                    x49=x45 + x48
                                    x50=x45 + 1
                                    
                                    # calculation with additional factor (1+(-1)**T) for two sided plate silencer
                                    Sum += (1+(-1)**T)*((1/2)*1j*c*rho0*(2 - deltas)*(2 - deltat)*(-K*N*x22*(-k0 + x6)*(k0 - x6)*((-1)**(S + x10+0j) + (-1)**(S + x9+0j) + x8 + 1)*((-1)**(x21+0j)*x11*(x12 - x19)*numpy.exp(-1j*x2) + x13 + x15*x17 - x15 + x18*x20)/((K + S)*(K + x0)*(N + S)*(N + x0)*(x1 + x2)*(x2 + x3)*(x2 + x4)*(x2 + x5)) + (1/2)*x22*x23*((N)**(2+0j) - 2*(S)**(2+0j))*(Kstm*M + k0)**(2+0j)*((-1)**(x44+0j)*x20 + (-1)**(x46+0j)*x20 + (-1)**(x10 + x41+0j)*x13 + (-1)**(x21 + x45+0j)*x13 + (-1)**(x21 + x49+0j)*x20 + (-1)**(x34 + x40+0j)*x20 + (-1)**(x41 + x7+0j)*x13 + (-1)**(x41 + x9+0j)*x13 + (-1)**(x44 + x9+0j)*x20 + (-1)**(x46 + x48+0j)*x13 + x11*x42*x43 + x11*((-1)**(K+0j)*x12 + (-1)**(N+0j)*x12 + (-1)**(x10+0j)*x19 + (-1)**(x45+0j)*x12 + (-1)**(x49+0j)*x19 + (-1)**(x50+0j)*x19 + (-1)**(x9+0j)*x19 + (-1)**(J + 1+0j)*x19*x47 + (-1)**(N + x9+0j)*x12 + (-1)**(x48 + x50+0j)*x12 + x12*x23*x47 + x19*x8 + x27 - x43)*numpy.exp(-1j*x26) + x13*x18*x47 + x17*x20*x47 - x23*x28 + x23*x29 + x28*x30 + x28*x32 + x28*x33 - x28*x35 - x28*x37 - x28*x38 + x28*x39 - x28*x42 - x29*x30 - x29*x32 - x29*x33 + x29*x35 + x29*x37 + x29*x38 - x29*x39)/(K*N*x25*(N - x24)*(x1 + x26)*(x26 + x3)*(x26 + x4)*(x26 + x5)))/(depth*height_d*(-k0**2 + kappast**2*(1 - M**2))**(1/2+0j)))
                                
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
    
    # method to calculate the transmission coefficient of the plate silencer
    def transmission(self, vp, height_d, I, M, medium, freq):
        
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
        tau = np.abs(np.sum(vp*(2*temp), axis=(0,1))+1)**2
        
        return tau
    
    # method to calculate the reflection coefficient of the plate silencer
    def reflection(self, vp, height_d, I, M, medium, freq):
        
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
        beta_temp = np.abs(np.sum(vp*(2*temp), axis=(0,1)))**2
        
        # energetic correction of reflection coefficient due to mean flow
        beta = beta_temp*((1-M)/(1+M))**2
        
        return beta
    
    # method to calculate the absorption coefficient of the plate silencer
    def absorption(self, vp, height_d, I, M, medium, freq):
        
        # transmission coefficient
        tau = self.transmission(vp, height_d, I, M, medium, freq)
        
        # reflection coefficient
        beta = self.reflection(vp, height_d, I, M, medium, freq)
        
        # absorption coefficient
        alpha = 1-tau-beta
        
        return alpha
    
    # method calculate the transmission loss of the plate silencer
    def transmissionloss(self, vp, height_d, I, M, medium, freq):
        
        # transmission coefficient
        tau = self.transmission(vp, height_d, I, M, medium, freq)
        
        # transmission loss
        TL = -10*np.log10(tau)
        
        return TL
    
