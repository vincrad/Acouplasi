#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:02:55 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from numpy import sqrt

# =============================================================================
# class Fluid(tr.HasTraits):
#     '''
#     Class to define the fluid in the duct.
#     '''
#     
#     rho0 = tr.Float()
#     kappa = 1.4
#     R = 287.058     #J/kgK
#     
#     #c = tr.property
#     
#     def soundspeed(self, temp):
#         
#         self.c = np.sqrt(self.kappa*self.R*(temp+273.15))
#         
#         return self.c
# =============================================================================
    
class Fluid(tr.HasTraits):
    '''
    Class to define the fluid in the duct.
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
    
        
    
    