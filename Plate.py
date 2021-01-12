#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:34:23 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from Material import Material


class Plate(tr.HasTraits):
    '''
    Class to define a plate for a plate resonator lining.
    '''
    
    hp = tr.Float()
    
    material = tr.Instance(Material)
    
    def lmatrix(self, length, depth, J, L, freq):
        
        # circular frquency
        omega = 2*np.pi*freq
        
        # area density
        m = self.material.mass(length)
        
        # bending stiffness
        B = self.material.bendingstiffness(self.hp, length)
        
        # calculation of L-Matrix
        Lmatrix = np.zeros((len(J), len(L), len(freq)), dtype=complex)
        
        for j in J:
        
            for l in L:
                
                if j==l:
            
                    Lmatrix[j-1,l-1,:] = ((B/1j*omega)*((l*np.pi)/length)**4+1j*omega*m)*(length/2)
                    
                else:
                    
                    pass
            
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
    