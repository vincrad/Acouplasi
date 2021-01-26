#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:18:56 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from Temperature import Temperature

class Material(tr.HasTraits):
    '''
    Class to define the material of the plate.
    '''
    
    # density of the plate 
    rhop = tr.Float()
    
    # temperature
    temperature = tr.Instance(Temperature)
    
    def mass(self, length):
        
        return self.rhop*length
        
    
    def bendingstiffness(self, hp, depth):
        
        E = 1000*10**(6) # Pa
        
        eta = 0.1
        
        Ecomplex = E*(1+1j*eta)
        
        I = (depth*hp**3)/12
        
        B = E*I
        
        return B
    
# =============================================================================
#      def bendingstiffness(self, freq):
#          
#          pass
# =============================================================================
