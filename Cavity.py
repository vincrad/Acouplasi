#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:49:08 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from Fluid import Fluid

class Cavity(tr.HasTraits):
    '''
    Class to define a cavity for a plate resonator lining.
    '''
    # height of cavity
    hc = tr.Float()
    
    fluid = tr.Instance(Fluid)
    
    def modes(self, length, depth, Lc, Mc, Nc):
        # Parameter müssen aus PlateResonatorLining bekannt sein
        # function which returns the modal sound field of the cavity
        pass