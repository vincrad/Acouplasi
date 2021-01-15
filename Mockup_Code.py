#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 15:27:37 2021

@author: radmann
"""

#%% Mockup-Code for Silencer

import numpy as np
import traitlets as tr

#%% Fluid

class Fluid(tr.HasTraits):
    
    '''
    Class to define a fluid in the duct.
    '''
    
    rho0 = tr.Float(1.2)
    kappa = tr.Float(1.4)
    R = tr.Float(287.058)   # J/kgK
    c = tr.Float()
    
    temperatureC = tr.Float(20)
    
    @property
    def c(self):
        
        c = np.sqrt(self.kappa*self.R*(self.temperatureC+273.15))
        
        return c


#%% Material

class Material(tr.HasTraits):
    
    '''
    Class to define the material of plate for a plate resonator silencer.
    '''
    
    # density
    rhop = tr.Float()
    
    # temperature
    # Übergabe der Temperatur evtl. anders lösen
    temperatureC = tr.Float(default_value=20)
    
    # method to calculate the area density of the plate
    def areadensity(self, length):
        
        m = np.random.random()
        
        return m
        
    # method to calculate the bending stiffness
    def bendingstiffness(self, hp, depth):
        
        B = np.random.random()
        

        return B



#%% Cavity

class Cavity(tr.HasTraits):
    
    '''
    Class to define a cavity for a plate resonator silencer.
    '''
    
    # height of the cavity
    height = tr.Float()
    
    # modes of the cavity
    R = tr.Instance(np.ndarray)
    S = tr.Instance(np.ndarray)
    
    # loss factor
    zetars = tr.Float(default_value=0.1)
    
    # medium
    medium = tr.Instance(Fluid)
    
    # method calculates kappe for impedance calculation
    def kappars(self):
        
        kappa = np.random.rand(len(self.R), len(self.S))
        
        return kappa
    
    # method calculates the cavity impedance matrix
    def cavityimpedance(self, length, depth, J, L, freq):
        
        omega = 2*np.pi*freq
        
        Zc = np.random.rand(len(J), len(L), len(freq))
        
        return Zc
    
    




#%% Plate

class Plate(tr.HasTraits):
    
    '''
    Class to define a plate for a plate resonator silencer.
    '''
    
    # plate height
    hp = tr.Float()
    
    # material of the plate
    material = tr.Instance(Material)
    
    # method calculates the L-Matrix for a plate resonator silencer
    def lmatrix(self, length, depth, J, L, freq):
        
        # circular frquency
        omega = 2*np.pi*freq
        
        # area density
        m = self.material.areadensity(length)
        
        # bending stiffness
        B = self.material.bendingstiffness(self.hp, length)
        
        # calculation of L-Matrix
        Lmatrix = np.random.rand(len(J), len(L), len(freq))
        
        return Lmatrix




#%% Linings

class Linings(tr.HasTraits):
    
    '''
    Parental class for different kinds of linings.
    '''
    
    # geometry
    length = tr.Float()
    depth = tr.Float(default_value=0)
    
class PlateResonators(Linings):
    
    '''
    Parental class for differnt kinds of plate resonator configurations.
    '''
    
    # plate modes
    J = tr.Instance(np.ndarray)
    L = tr.Instance(np.ndarray)
    
class SinglePlateResonator(PlateResonators):
    
    '''
    2D single plate resonator without flow.
    '''
    
    # plate
    plate = tr.Instance(Plate)
    
    # cavity
    cavity = tr.Instance(Cavity)
    
    # method calculates the impedance matrix of the plate resonator
    def zmatrix(self, freq, M):
        
        # cicular frequency and wave number
        omega = 2*np.pi*freq
        k0 = omega/self.cavity.medium.c
        
        # cavity impedance matrix
        Zc = self.cavity.cavityimpedance(self.length, self.depth, self.J, self.L, freq)
        
        # impedance matrix of plate induced sound
        Zprad = np.random.rand(len(self.J), len(self.L), len(freq))
        
        Z = Zc+Zprad
        
        return Z
    
    # method calculates the platevelocity
    def platevelocity(self, I, M, freq):
        
        # impedance matrix
        Z = self.zmatrix(freq, M)
        
        # L-Matrix with material properties of the plate
        Lmatrix = self.plate.lmatrix(self.length, self.depth, self.J, self.L, freq)
        
        # solving the linear system of equation
        
        # building lhs matrix from Z and Lmatrix
        lhs = Z+Lmatrix
        
        # plate velocity array
        # Lösung des LGS enthält I
        vp = np.random.rand(len(self.L), len(freq))
        
        return vp
    
    # method calculates the transmission loss of the plate silencer
    def transmissionloss(self, I, M, freq):
        
        # plate velocity
        vp = self.platevelocity(I, M, freq)
        
        TL = vp*2
        
        return TL
        
class ReflectionLining(Linings):
    
    '''
    Class calculates kz and Z for a reflection silencer.
    '''
    # geometry
    height = tr.Float()
    
    @property
    def S(self):
        
        S = self.depth*self.height
        
        return S
    
    # method calculates kz and Z
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
        
        S = self.depth*self.height
        
        return S
    
    def Zkz(self, medium, freq):
        
        pass
        # return (kz, Z)
    
    
        
#%% DuctElement

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
    
    # method calculates the incident sound array for plate silencer
    def incidentsound(self, freq):
        
        omega = 2*np.pi*freq
        
        k0 = omega/self.medium.c
        
        I = np.random.rand(len(self.lining.L), len(omega))
        
        return I
    
    # method calculates the transfermatrix for reflection and absorption silener
    # or the transmission loss for plate silencer
    def tmatrix_tl(self, freq):
        
        if isinstance(self.lining, PlateResonators)==True:
            
            I = self.incidentsound(freq)
            
            TL = self.lining.transmissionloss(I, self.M, freq)
            
            return TL
            
        else:
            
            def T(self,freq):
        
                [kz, Z] = self.lining.Zkz(self.medium, freq)
   
                T = np.array([[np.cos(kz*self.lining.length), 1j*Z*np.sin(kz*self.lining.length)],[1j*(1/Z)*np.sin(kz*self.lining.length), np.cos(kz*self.lining.length)]])
        
            return T
            
       
    



#%% Duct-Class

class Duct(tr.HasTraits):
    
    '''
    Combines different DuctElements to build a silencer.
    '''
    # frequency
    freq = tr.Instance(np.ndarray)
    
    # list of DuctElements
    elements = tr.List(trait = tr.Instance(DuctElement))
    
    # method calculates the transmission loss of the duct
    def tl(self):
        
        for i in self.elements:
            
            TL = i.tmatrix_tl(self.freq)
            
        return TL
            
            #if isinstance(i, PlateResonators)==True:
                
                #TL = i.tmatrix_tl(self.freq)
                #print(TL)
                
                
                
        
        
        
    
    
