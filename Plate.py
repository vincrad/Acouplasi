#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:34:23 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from Material import Material
from Temperature import Temperature

#%%

class Plate(tr.HasTraits):
    
    '''
    Parental class to define a singel, double or triple layer plate for a 2D plate silencer.
    '''
    
    # temperature
    temperature = tr.Instance(Temperature)

class SimplePlate(Plate):
    
    '''
    Class to define a single layer plate for a plate resonator silencer.
    '''
    
    # plate height
    hp = tr.Float
    
    # material of the plate
    material = tr.Instance(Material)
    

    
    # mass per area
    def mass(self):
                   
        return self.material.rho*self.hp
    
    # bendingstiffness
    def bendingstiffness(self, freq, temp):
        
        return self.material.E(freq,temp)*self.hp**3/12/(1-self.material.mu**2)
    
    # method calculate the L matrix for a plate resonator silencer
    def lmatrix(self, length, depth, l, freq):
        
        # define extended array of modes
        L = l[:, np.newaxis]
        
        # circular frequency
        omega = 2*np.pi*freq
        
        # calculate area density
        m = self.mass()
        
        # calculate bending stiffness
        B = self.bendingstiffness(freq, self.temperature)
        
        # calculate the values of the L matrix of the plate
        Lmatrix_temp = (B/(1j*omega)*((L*np.pi)/length)**4+1j*omega*m)*(length/2)
        
        # diagonalize the L matrix
        Lmatrix = Lmatrix_temp*np.expand_dims(np.identity(len(l)), 2)
        
        return Lmatrix
        

    
class DoubleLayerPlate(Plate):
    
    '''
    Class to define a double layer plate for a plate resonator silencer.
    '''
    
    # plate heights
    hp1 = tr.Float
    hp2 = tr.Float
    
    # materials of the plates
    material1 = tr.Instance(Material)
    material2 = tr.Instance(Material)
    
    # mass per area
    def mass(self):
                   
        return self.material1.rho*self.hp1+self.material2.rho*self.hp2
    
    # bendingstiffness
    def bendingstiffness(self, freq, temp):
        
        return (self.material1.E(freq,temp)*self.hp1**3/12/(1-self.material1.mu**2)
                +self.material2.E(freq,temp)*self.hp2**3/12/(1-self.material2.mu**2)
                +(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)*self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2))
                /(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)+self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2))
                *(self.hp1+self.hp2)**2/4)
    
    # method calculate the L matrix for a plate resonator silencer
    def lmatrix(self, length, depth, l, freq):
        
        # define extended array of modes
        L = l[:, np.newaxis]
        
        # circular frequency
        omega = 2*np.pi*freq
        
        # calculate area density
        m = self.mass()
        
        # calculate bending stiffness
        B = self.bendingstiffness(freq, self.temperature)
        
        # calculate the values of the L matrix of the plate
        Lmatrix_temp = (B/(1j*omega)*((L*np.pi)/length)**4+1j*omega*m)*(length/2)
        
        # diagonalize the L matrix
        Lmatrix = Lmatrix_temp*np.expand_dims(np.identity(len(l)), 2)
        
        return Lmatrix
    
class TripleLayerPlate(Plate):
    
    '''
    Class to define a triple layer plate for a plate resonator silencer.
    '''
    
    # plate heights
    hp1 = tr.Float
    hp2 = tr.Float
    hp3 = tr.Float
    
    # materials of the plates
    material1 = tr.Instance(Material)
    material2 = tr.Instance(Material)
    material3 = tr.Instance(Material)
    
    # mass per area
    def mass(self):
                   
        return self.material1.rho*self.hp1+self.material2.rho*self.hp2+self.material3.rho*self.hp3
    
    # bendingstiffness
    def bendingstiffness(self, freq, temp):
        
        return (self.material1.E(freq,temp)*self.hp1**3/12/(1-self.material1.mu**2)
               +self.material2.E(freq,temp)*self.hp2**3/12/(1-self.material2.mu**2)
               +self.material3.E(freq,temp)*self.hp3**3/12/(1-self.material3.mu**2)
               +((self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)*self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2))*(self.hp1+self.hp2)**2
                +(self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2)*self.material3.E(freq,temp)*self.hp3/(1-self.material3.mu**2))*(self.hp2+self.hp3)**2
                +(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)*self.material3.E(freq,temp)*self.hp3/(1-self.material3.mu**2))*(self.hp1+self.hp3)**2)
                /(4*(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)
                    +self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2)
                    +self.material3.E(freq,temp)*self.hp3/(1-self.material3.mu**2)))
               +(self.material1.E(freq,temp)/(1-self.material1.mu**2)*self.material3.E(freq,temp)/(1-self.material3.mu**2)
                 *self.hp1*self.hp2*self.hp3*(self.hp1+self.hp2+self.hp3))
                /(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)
                 +self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2)
                 +self.material3.E(freq,temp)*self.hp3/(1-self.material3.mu**2)))
    
    
    # method calculate the L matrix for a plate resonator silencer
    def lmatrix(self, length, depth, l, freq):
        
        # define extended array of modes
        L = l[:, np.newaxis]
        
        # circular frequency
        omega = 2*np.pi*freq
        
        # calculate area density
        m = self.mass()
        
        # calculate bending stiffness
        B = self.bendingstiffness(freq, self.temperature)
        
        # calculate the values of the L matrix of the plate
        Lmatrix_temp = (B/(1j*omega)*((L*np.pi)/length)**4+1j*omega*m)*(length/2)
        
        # diagonalize the L matrix
        Lmatrix = Lmatrix_temp*np.expand_dims(np.identity(len(l)), 2)
        
        return Lmatrix
    
#%%

class Plate3D(tr.HasTraits):
    
    '''
    Parental class to define a single, double or triple layer plate for a 3D plate silencer.
    '''
    
    # temperature
    temperature = tr.Instance(Temperature)
    
class SimplePlate3D(Plate3D):
    
    '''
    Class to define a single layer plate for 3D plate silencer.
    '''
    # plate height
    hp = tr.Float()
    
    # material of the plate
    material = tr.Instance(Material)
    
    # mass per area
    def mass(self):
        
        return self.material.rho*self.hp
    
    # beding stiffness
    def bendingstiffness(self, freq, temp):
        
        return self.material.E(freq, temp)*self.hp**3/12/(1-self.material.mu**2)
    
    # method calculate the L matrix of the 3D plate silencer
    def lmatrix(self, length, depth, l, n, freq):
        
        # define extended arrays of modes
        L = l[:, np.newaxis, np.newaxis]
        N = n[np.newaxis, :, np.newaxis]
        
        # circular frequency
        omega = 2*np.pi*freq
        
        # calculate area density
        m = self.mass()
        
        # calculate bending stiffness
        B = self.bendingstiffness(freq, self.temperature)
        
        # calculate the values of L matrix of the plate
        Lmatrix_temp = ((B/(1j*omega))*(((L*np.pi)/length)**4+2*((L*np.pi)/length)**2*((N*np.pi)/depth)**2+((N*np.pi)/depth)**4)+1j*omega*m)*((length*depth)/4)
        
        # expand L matrix taking into account jk=lm
        # building an array with 1 for jk=lm and 0 for jk!=lm
        j,k = np.ogrid[:len(l), :len(l)]
        kron = np.zeros((len(l), len(l), len(l), len(l)))[:, :, :, :, np.newaxis]
        kron[j,k,j,k] = 1
        
        # expand L matrix
        Lmatrix = Lmatrix_temp*kron

        return Lmatrix

class DoubleLayerPlate3D(Plate):
    
    '''
    Class to define a double layer plate for a 3D plate silencer.
    '''
    
    # plate heights
    hp1 = tr.Float()
    hp2 = tr.Float()
    
    # material of the plates
    material1 = tr.Instance(Material)
    material2 = tr.Instance(Material)
    
    # mass per area
    def mass(self):
                   
        return self.material1.rho*self.hp1+self.material2.rho*self.hp2
    
    # bendingstiffness
    def bendingstiffness(self, freq, temp):
        
        return (self.material1.E(freq,temp)*self.hp1**3/12/(1-self.material1.mu**2)
                +self.material2.E(freq,temp)*self.hp2**3/12/(1-self.material2.mu**2)
                +(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)*self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2))
                /(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)+self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2))
                *(self.hp1+self.hp2)**2/4)
        
    # method calculate the L matrix of the 3D plate silencer
    def lmatrix(self, length, depth, l, n, freq):
        
        # define extended arrays of modes
        L = l[:, np.newaxis, np.newaxis]
        N = n[np.newaxis, :, np.newaxis]
        
        # circular frequency
        omega = 2*np.pi*freq
        
        # calculate area density
        m = self.mass()
        
        # calculate bending stiffness
        B = self.bendingstiffness(freq, self.temperature)
        
        # calculate the values of L matrix of the plate
        Lmatrix_temp = ((B/(1j*omega))*(((L*np.pi)/length)**4+2*((L*np.pi)/length)**2*((N*np.pi)/depth)**2+((N*np.pi)/depth)**4)+1j*omega*m)*((length*depth)/4)
        
        # expand L matrix taking into account jk=lm
        # building an array with 1 for jk=lm and 0 for jk!=lm
        j,k = np.ogrid[:len(l), :len(l)]
        kron = np.zeros((len(l), len(l), len(l), len(l)))[:, :, :, :, np.newaxis]
        kron[j,k,j,k] = 1
        
        # expand L matrix
        Lmatrix = Lmatrix_temp*kron
        
        return Lmatrix
    
class TripleLayerPlate3D(Plate3D):
    
    '''
    Class to define a triple layer plate for a 3D plate silencer.
    '''
    
    # plate heights
    hp1 = tr.Float()
    hp2 = tr.Float()
    hp3 = tr.Float()
    
    # materials of the plates
    material1 = tr.Instance(Material)
    material2 = tr.Instance(Material)
    material3 = tr.Instance(Material)
    
    # mass per area
    def mass(self):
                   
        return self.material1.rho*self.hp1+self.material2.rho*self.hp2+self.material3.rho*self.hp3
    
    # bendingstiffness
    def bendingstiffness(self, freq, temp):
        
        return (self.material1.E(freq,temp)*self.hp1**3/12/(1-self.material1.mu**2)
               +self.material2.E(freq,temp)*self.hp2**3/12/(1-self.material2.mu**2)
               +self.material3.E(freq,temp)*self.hp3**3/12/(1-self.material3.mu**2)
               +((self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)*self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2))*(self.hp1+self.hp2)**2
                +(self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2)*self.material3.E(freq,temp)*self.hp3/(1-self.material3.mu**2))*(self.hp2+self.hp3)**2
                +(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)*self.material3.E(freq,temp)*self.hp3/(1-self.material3.mu**2))*(self.hp1+self.hp3)**2)
                /(4*(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)
                    +self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2)
                    +self.material3.E(freq,temp)*self.hp3/(1-self.material3.mu**2)))
               +(self.material1.E(freq,temp)/(1-self.material1.mu**2)*self.material3.E(freq,temp)/(1-self.material3.mu**2)
                 *self.hp1*self.hp2*self.hp3*(self.hp1+self.hp2+self.hp3))
                /(self.material1.E(freq,temp)*self.hp1/(1-self.material1.mu**2)
                 +self.material2.E(freq,temp)*self.hp2/(1-self.material2.mu**2)
                 +self.material3.E(freq,temp)*self.hp3/(1-self.material3.mu**2)))
        
    # method calculate the L matrix of the 3D plate silencer
    def lmatrix(self, length, depth, l, n, freq):
        
        # define extended arrays of modes
        L = l[:, np.newaxis, np.newaxis]
        N = n[np.newaxis, :, np.newaxis]
        
        # circular frequency
        omega = 2*np.pi*freq
        
        # calculate area density
        m = self.mass()
        
        # calculate bending stiffness
        B = self.bendingstiffness(freq, self.temperature)
        
        # calculate the values of L matrix of the plate
        Lmatrix_temp = ((B/(1j*omega))*(((L*np.pi)/length)**4+2*((L*np.pi)/length)**2*((N*np.pi)/depth)**2+((N*np.pi)/depth)**4)+1j*omega*m)*((length*depth)/4)
        
        # expand L matrix taking into account jk=lm
        # building an array with 1 for jk=lm and 0 for jk!=lm
        j,k = np.ogrid[:len(l), :len(l)]
        kron = np.zeros((len(l), len(l), len(l), len(l)))[:, :, :, :, np.newaxis]
        kron[j,k,j,k] = 1
        
        # expand L matrix
        Lmatrix = Lmatrix_temp*kron
        
        return Lmatrix