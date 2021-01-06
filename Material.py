#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 13:18:56 2020

@author: radmann
"""

import traitlets as tr
import numpy as np

class Material(tr.HasTraits):
    '''
    Class to define the material of the plate.
    '''
    
    def mass(self, rhop):
        
        pass
    
    def bendingstiffness(self, temperatureC, freq):
        
        pass