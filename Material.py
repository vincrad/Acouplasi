#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created February 2021

@authors: radmann, schwaericke
"""

import traitlets as tr
import numpy as np

#%%

class Material(tr.HasTraits):
    '''
    Class to define the material of the plate.
    ''' 
    # density
    rho = tr.Float()
    
    # Complex Youngs Modulus
    E = tr.Callable()

    # Poissons Ratio
    mu = tr.Float()


# Polyurethane TPU1195A
TPU1195A = Material(rho = 1150, mu = .48, E = lambda freq, temp: 3.1e8*.97**(temp.C-7*np.log10(freq))+1j*4.7e7*.96**(temp.C-7*np.log10(freq)))

# Lead
Pb = Material(rho = 11300, mu = .43, E = lambda freq, temp: 1.7e10*(1+1j*.1))

# Copper
Cu = Material(rho = 8900, mu = .35, E = lambda freq, temp: 1.25e11*(1+1j*.002))

# Steel
Steel = Material(rho = 7800, mu = .31, E = lambda freq, temp: 2.1e11*(1+1j*.0003))

# Aluminium
Al = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))

# Magnesium
Mg = Material(rho = 1740, mu = .29, E = lambda freq, temp: 4.31e10*(1+1j*.0001))

# Polyurethane TPU1170A at 25°C 
TPU1170A = Material(rho = 1080, mu = .48, E = lambda freq, temp: 1.4e7*(1+1j*.05))

# Polymere KKP07
KKP07 = Material(rho = 1100, mu = .45, E = lambda freq, temp: 2.68e7*(1+1j*.55))
