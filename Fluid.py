#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:02:55 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from numpy import sqrt

class Fluid(tr.HasTraits):
    '''
    Class to define the fluid in the duct.
    '''
    
    rho0 = tr.Float()
    kappa = 1.4
    R = 287.058     #J/kgK
    
    def soundspeed(self, temp):
        
        self.c = np.sqrt(self.kappa*self.R*(temp+273.15))
        
        return self.c
    
# =============================================================================
#     rhof = tr.Float()
#     R = tr.Float()
#     kappa = tr.Float()
#     T = tr.List(trait = tr.Float()) 
#     
#     def soundspeed(self):
#         T = np.asarray(self.T)
#         c = sqrt(self.kappa*self.R*T)
#         return c
# =============================================================================

    
    