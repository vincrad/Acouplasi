#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:27:57 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from Plate import Plate
from Cavity import Cavity

class StandardPlateResonator(tr.HasTraits):
    '''
    Standard plate resonator configuration. Two opposite single plates.
    '''
    
    # calling up the plate
    plate = tr.Instance(Plate)
    
    # calling up the cavity
    cavity = tr.Instance(Cavity)
    
    

    