#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 07:35:22 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from DuctElement import DuctElement
import matplotlib.pyplot as plt
from numpy import log10
from numpy import abs
 
#%%
       
class Duct(tr.HasTraits):

    '''
    Combines different DuctElements to build a silencer.
    '''
    
    # frequency
    freq = tr.Instance(np.ndarray)
    
    # duct height
    height_d = tr.Float()

    # list of DuctElements
    elements = tr.List(trait = tr.Instance(DuctElement))
    
    # method calculate teh transmission loss of the duct
    def tl(self):
        
        for i in self.elements:
            
            TL = i.tmatrix(self.height_d, self.freq)
            
            return TL
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
      


    

