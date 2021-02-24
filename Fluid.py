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
    
    p0 = tr.Float(101325)
    rho0 = tr.Float()
    kappa = tr.Float(1.4)
    R = tr.Float(287.058)   # J/kgK
    c = tr.Float()
    
    temperature = tr.Instance(Temperature)
    
    @property
    def c(self):
        
        return np.sqrt(self.kappa*self.R*(self.temperature.C+273.15))
    
    @property
    def rho0(self):
        
        return self.p0/(self.R*(self.temperature.C+273.15))
