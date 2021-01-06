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
    
    rhop = tr.Float()
    
    mat = tr.Instance(Material)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

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
    