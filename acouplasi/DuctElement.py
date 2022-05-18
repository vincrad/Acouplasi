#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 09:47:36 2020

@author: radmann
"""

import traitlets as tr
import numpy as np, numpy
from .Fluid import Fluid
from .Linings import Linings, NoLining, NoLining3D, PlateResonators, SinglePlateResonator, SinglePlateResonator3D

#%%

class DuctElement(tr.HasTraits):
    
    '''
    Class calculates the transfer matrix of a certain duct section.
    '''
    
    # lining of the duct element
    lining = tr.Instance(Linings)
    
    # fluid
    medium = tr.Instance(Fluid)
    
    # flow
    M = tr.Float(default_value=0)
    
    # method calculate the incident sound array for plate silencer
    def incidentsound(self, M, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c
        
        # define expended array of modes
        L = self.lining.l[:, np.newaxis]
        
        # calculate incident sound array
        x0=self.lining.length**2*k0**2
        x1=numpy.pi**2*L**2
        x2=2*M
        x3=M**2
        x4=x1*x3
        x5=numpy.pi*self.lining.length*L
        x6=x5/(x0 - x1*x2 - x1 - x4)
        x7=numpy.exp(1j*self.lining.length*k0/(M + 1))
        x8=x1*x7
        x9=(-1)**L*x5/(x0*x7 - x2*x8 - x4*x7 - x8)
        
        I = self.medium.c**2*self.medium.rho0*(-x2*x6 + x2*x9 - x3*x6 + x3*x9 - x6 + x9)
        
        return I
    
    # method calculates the scattering matrix of the duct element
    def scatteringmatrix(self, height_d, freq):
        
        # plate resonator silencer
        if isinstance(self.lining, PlateResonators)==True:
            
            # incident sound
            I = self.incidentsound(self.M, freq)
            
            # plate velocity
            vp = self.lining.platevelocity(height_d, I, self.M, freq)
            
            # transmission factor
            tra_fac = self.lining.transmissionfactor(vp, height_d, I, self.M, self.medium, freq)
            
            # reflection factor
            ref_fac = self.lining.reflectionfactor(vp, height_d, I, self.M, self.medium, freq)
            
            # scattering matrix
            SM = np.array([[ref_fac, tra_fac], [tra_fac, ref_fac]])
            
            return SM
        
        elif isinstance(self.lining, NoLining)==True:
            
            # transmission factor
            tra_fac = self.lining.transmissionfactor(self.M, self.medium, freq)
            
            # reflection factor
            ref_fac = self.lining.reflectionfactor(self.M, self.medium, freq)
            
            # scattering matrix
            SM = np.array([[ref_fac, tra_fac], [tra_fac, ref_fac]])
            
            return SM
        
        # reflection and absorption silencer
        else:
            
            # calculate Z and kz for absorption or reflection lining
            [kz, Z] = self.lining.Zkz(self.medium, height_d, freq)

            # calculate the transfer matrix for the certain duct element
            TM = np.array([[np.cos(kz*self.lining.length), 1j*Z*np.sin(kz*self.lining.length)],[1j*(1/Z)*np.sin(kz*self.lining.length), np.cos(kz*self.lining.length)]])
            
            # calculate the scattering matrix from transfer matrix
            # admittance matrix
            YM = np.empty((2,2,len(freq)), dtype=complex)
            
            YM[0,0,:] = TM[1,1,:]/TM[0,1,:]
            YM[0,1,:] = TM[1,0,:]-((TM[0,0,:]*TM[1,1,:])/(TM[0,1,:]))
            YM[1,0,:] = -1/TM[0,1,:]
            YM[1,1,:] = TM[0,0,:]/TM[0,1,:]
            
            # scattering matrix of the duct element
            Z0 = self.medium.rho0*self.medium.c
            E = np.eye(2, dtype=complex)
            
            SM = np.empty((2,2,len(freq)), dtype=complex)
            
            for idx, item in enumerate(freq):
                
                SM[:,:,idx] = (E-Z0*YM[:,:,idx]) @ np.linalg.inv(E+Z0*YM[:,:,idx])
            
            return SM
            
            
            
        
    # method calculates the transmission coefficients, reflection coefficient and dissipation coefficient from the scattering matrix of the duct element
    def scatteringcoefficients(self, height_d, freq):
        
        # scattering matrix
        SM = self.scatteringmatrix(height_d, freq)
        
        # transmission coefficient
        tra = np.abs(SM[1,0,:])**2
        
        # reflaction coefficient
        ref = np.abs(SM[0,0,:])**2*((1-self.M)/(1+self.M))**2
        
        # dissipation coefficient
        dis = 1-tra-ref
        
        return tra, ref, dis
        
#%%

class DuctElement3D(tr.HasTraits):
    
    '''
    Class calculates the transfer matrix of a certain 3D duct section.
    '''
    
    # lining of the duct element
    lining = tr.Instance(Linings) # anpassen 2D/3D
    
    # fluid
    medium = tr.Instance(Fluid)
    
    # flow
    M = tr.Float(default_value=0)
    
    # method calculate the incident sound array for plate silencer
    def incidentsound(self, M, freq):
        
        # circular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.medium.c

        # define expended arrays of modes
        L = self.lining.l[:, np.newaxis, np.newaxis]
        N = self.lining.n[np.newaxis, :, np.newaxis]
        
        # calculate incident sound array
        x0=2*M
        x1=(M)**(2+0j)
        x2=1j*self.lining.length*k0/(M + 1)
        x3=(np.pi)**(2+0j)*(L)**(2+0j)
        
        I = -self.lining.length*self.lining.depth*(self.medium.c)**(2+0j)*L*self.medium.rho0*((-1)**(N+0j) - 1)*((-1)**(L + 1+0j) + numpy.exp(x2))*(x0 + x1 + 1)*numpy.exp(-x2)/(N*(-(self.lining.length)**(2+0j)*(k0)**(2+0j) + x0*x3 + x1*x3 + x3))
        
        return I
    
        # method calculates the scattering matrix of the duct element
    def scatteringmatrix(self, height_d, freq):
        
        # plate resonator silencer
        if isinstance(self.lining, PlateResonators)==True:
            
            # incident sound
            I = self.incidentsound(self.M, freq)
            
            # plate velocity
            vp = self.lining.platevelocity(height_d, I, self.M, freq)
            
            # transmission factor
            tra_fac = self.lining.transmissionfactor(vp, height_d, I, self.M, self.medium, freq)
            
            # reflection factor
            ref_fac = self.lining.reflectionfactor(vp, height_d, I, self.M, self.medium, freq)
            
            # scattering matrix
            SM = np.array([[ref_fac, tra_fac], [tra_fac, ref_fac]])
            
            return SM
        
        elif isinstance(self.lining, NoLining3D)==True:
            
            # transmission factor
            tra_fac = self.lining.transmissionfactor(self.M, self.medium, freq)
            
            # reflection factor
            ref_fac = self.lining.reflectionfactor(self.M, self.medium, freq)
            
            # scattering matrix
            SM = np.array([[ref_fac, tra_fac], [tra_fac, ref_fac]])
            
            return SM        
        
        # reflection and absorption silencer
        else:
            
            pass
        
    # method calculates the transmission coefficients, reflection coefficient and dissipation coefficient from the scattering matrix of the duct element
    def scatteringcoefficients(self, height_d, freq):
        
        # scattering matrix
        SM = self.scatteringmatrix(height_d, freq)
        
        # transmission coefficient
        tra = np.abs(SM[1,0,:])**2
        
        # reflaction coefficient
        ref = np.abs(SM[0,0,:])**2*((1-self.M)/(1+self.M))**2
        
        # dissipation coefficient
        dis = 1-tra-ref
        
        return tra, ref, dis