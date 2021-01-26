#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 12:02:55 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from numpy import sqrt
from Temperature import Temperature

    
class Fluid(tr.HasTraits):
    '''
    Class to define the fluid in the duct.
    '''
    
    rho0 = tr.Float(1.2)
    kappa = tr.Float(1.4)
    R = tr.Float(287.058)   # J/kgK
    c = tr.Float()
    
    temperature = tr.Instance(Temperature)
    
    @property
    def c(self):
        
        return np.sqrt(self.kappa*self.R*(self.temperature.C+273.15))
        
    
        
    
    