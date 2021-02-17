#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:34:23 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from Material import Material


# =============================================================================
# class Plate(tr.HasTraits):
#     '''
#     Class to define a plate for a plate resonator lining.
#     '''
#     
#     hp = tr.Float()
#     
#     material = tr.Instance(Material)
#     
#     def lmatrix(self, length, depth, J, L, freq):
#         
#         # circular frquency
#         omega = 2*np.pi*freq
#         
#         # area density
#         m = self.material.mass(length)
#         
#         # bending stiffness
#         B = self.material.bendingstiffness(self.hp, length)
#         
#         # calculation of L-Matrix
#         Lmatrix = np.zeros((len(J), len(L), len(freq)), dtype=complex)
#         
#         for j in J:
#         
#             for l in L:
#                 
#                 if j==l:
#             
#                     Lmatrix[j-1,l-1,:] = ((B/1j*omega)*((l*np.pi)/length)**4+1j*omega*m)*(length/2)
#                     
#                 else:
#                     
#                     pass
#             
#         return Lmatrix
# =============================================================================
        
class Plate(tr.HasTraits):
    
    '''
    Class to define a plate for a plate resonator silencer.
    '''
    
    # plate height
    hp = tr.Float
    
    # material of the plate
    material = tr.Instance(Material)   
    
    # method calculate the L matrix for a plate resonator silencer
    def lmatrix(self, length, depth, l, freq):
        
        # define extended array of modes
        L = l[:, np.newaxis]
        
        # circular frequency
        omega = 2*np.pi*freq
        
        # calculate area density
        m = self.material.mass(self.hp)
        
        # calculate bending stiffness
        B = self.material.bendingstiffness(self.hp, freq)
        
        # calculate the values of the L matrix of the plate
        Lmatrix_temp = (B/(1j*omega)*((L*np.pi)/length)**4+1j*omega*m)*(length/2)
        
        # diagonalize the L matrix
        Lmatrix = Lmatrix_temp*np.expand_dims(np.identity(len(l)), 2)
        
        return Lmatrix
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# =============================================================================
# class Plate(tr.HasTraits):
#     '''
#     Class to define a plate for a plate resonator lining.
#     '''
#     
#     hp = tr.Float()     # plate thickness
#     rhop = tr.Float()   # Dichte der Platte
#     
#     def mass(self):
#         # zur Berechnung der Plattenmasse -> length and depth aus PlateResonatorLining
#        # mp = self.rhop*(self.hp*length*depth)
#        # return mp
#         pass
#     
#     def bendingstiffnes(self, freq, temp):
#         # calculation of bending stiffness with temperature and frequency dependend modulus of elasticity and loss factor
# =============================================================================
    