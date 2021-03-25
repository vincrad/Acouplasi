#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:22:29 2021

@author: radmann
"""

import traitlets as tr
import numpy as np

#%%

class Temperature(tr.HasTraits):
    
    '''
    Class to define the temperature in the duct.
    '''
    
    # temperature in Celsius
    C = tr.Float()
    

    