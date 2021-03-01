#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created February 2021

@authors: radmann, schwaericke
"""

import traitlets as tr
import numpy as np


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


    def mass(self, hp):
                   
        return self.rho*hp
    
    
    def bendingstiffness(self, hp, freq, temp):
        
        return self.E(freq,temp)*hp**3/12/(1-self.mu**2)

def E_func(freq,temp):
    if temp == -50:
        return (2.33e+09*freq**.108) + 1j*(1.08e8*freq**-0.027)
    elif temp == -25:
        return (9.54e+08*freq**.108) + 1j*(1.35e8*freq**-0.027)
    elif temp == 0:
        return (2.59e+08*freq**.158) + 1j*(7.06e7*freq**.081)
    elif temp == 25:
        return (7.86e+07*freq**.138) + 1j*(1.55e7*freq**.172)
    elif temp == 50:
        return (4.76e+07*freq**.069) + 1j*(4.39e6*freq**.149)
    else:
        return 1.6e8*(1+1j*.166)

# Polyurethane TPU1195A at 25°C
TPU1195A = Material(rho = 1150, mu = .48, E = E_func )

# Lead
Pb = Material(rho = 11300, mu = .43, E = lambda freq, temp: 1.7e10*(1+1j*.02))

# Copper
Cu = Material(rho = 8900, mu = .35, E = lambda freq, temp: 1.25e11*(1+1j*.002))

# Steel
Steel = Material(rho = 7800, mu = .31, E = lambda freq, temp: 2.1e11*(1+1j*.0002))

# Aluminium
Al = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))

# Magnesium
Mg = Material(rho = 1740, mu = .29, E = lambda freq, temp: 4.31e10*(1+1j*.0001))

# Polyurethane TPU1170A at 25°C 
TPU1170A = Material(rho = 1080, mu = .48, E = lambda freq, temp: 1.4e7*(1+1j*.05))

# Polymere KKP07
KKP07 = Material(rho = 1100, mu = .45, E = lambda freq, temp: 2.68e7*(1+1j*.55))
