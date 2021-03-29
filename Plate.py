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
    Class to define a plate for a plate resonator silencer.
    '''
    
    # plate height
    hp = tr.Float
    
    # material of the plate
    material = tr.Instance(Material)
    
    # temperature
    temperature = tr.Instance(Temperature)
    
    # method calculate the L matrix for a plate resonator silencer
    def lmatrix(self, length, depth, l, freq):
        
        # define extended array of modes
        L = l[:, np.newaxis]
        
        # circular frequency
        omega = 2*np.pi*freq
        
        # calculate area density
        m = self.material.mass(self.hp)
        
        # calculate bending stiffness
        B = self.material.bendingstiffness(self.hp, freq, self.temperature)
        
        # calculate the values of the L matrix of the plate
        Lmatrix_temp = (B/(1j*omega)*((L*np.pi)/length)**4+1j*omega*m)*(length/2)
        
        # diagonalize the L matrix
        Lmatrix = Lmatrix_temp*np.expand_dims(np.identity(len(l)), 2)
        
        return Lmatrix
        

class Plate3D(tr.HasTraits):
    
    '''
    Class to define a plate for a 3D plate silencer.
    '''
    
    # plate height
    hp = tr.Float()
    
    # material of the plate
    material = tr.Instance(Material)
    
    # temperature
    temperature = tr.Instance(Temperature)
    
    # method calculate the L matrix of the 3D plate silencer
    def lmatrix(self, length, depth, l, n, freq):
        
        # define extended arrays of modes
        L = l[:, np.newaxis, np.newaxis]
        N = n[np.newaxis, :, np.newaxis]
        
        # circular frequency
        omega = 2*np.pi*freq
        
        # calculate area density
        m = self.material.mass(self.hp)
        
        # calculate bending stiffness
        B = self.material.bendingstiffness(self.hp, freq, self.temperature)
        
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
