#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created February 2021

@authors: radmann, schwaericke
"""

import traitlets as tr
import numpy as np
#from Temperature import Temperature


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

    # temperature
#    temperature = tr.Instance(Temperature)
    
    def mass(self, hp):
                   
        return self.rho*hp
    
    
    def bendingstiffness(self, hp, freq):
        
        return self.E(freq)*hp**3/12/(1-self.mu**2)


# Polyurethane TPU1195A at -50°C frequency dependent
TPU1195A_m50 = Material(rho=1150, mu=.48, E = lambda freq: (2.33e+09*freq**.108) + 1j*(1.08e8*freq**-0.027))    
    
# Polyurethane TPU1195A at -25°C frequency dependent
TPU1195A_m25 = Material(rho=1150, mu=.48, E = lambda freq: (9.54e+08*freq**.108) + 1j*(1.35e8*freq**-0.027))    
    
# Polyurethane TPU1195A at 0°C frequency dependent
TPU1195A_0 = Material(rho=1150, mu=.48, E = lambda freq: (2.59e+08*freq**.158) + 1j*(7.06e7*freq**.081))

# Polyurethane TPU1195A at 25°C frequency dependent
TPU1195A_25 = Material(rho=1150, mu=.48, E = lambda freq: (7.86e+07*freq**.138) + 1j*(1.55e7*freq**.172))

# Polyurethane TPU1195A at 50°C frequency dependent
TPU1195A_50 = Material(rho=1150, mu=.48, E = lambda freq: (4.76e+07*freq**.069) + 1j*(4.39e6*freq**.149))

# Polyurethane TPU1195A at 25°C
TPU1195A = Material(rho = 1150, mu = .48, E = lambda freq: 1.6e8*(1+1j*.166))

# Lead
Pb = Material(rho = 11300, mu = .43, E = lambda freq: 1.7e10*(1+1j*.02))

# Copper
Cu = Material(rho = 8900, mu = .35, E = lambda freq: 1.25e11*(1+1j*.002))

# Steel
Steel = Material(rho = 7800, mu = .31, E = lambda freq: 2.1e11*(1+1j*.0002))

# Aluminium
Al = Material(rho = 2700, mu = .34, E = lambda freq: 7.21e10*(1+1j*.0001))

# Magnesium
Mg = Material(rho = 1740, mu = .29, E = lambda freq: 4.31e10*(1+1j*.0001))

# Polyurethane TPU1170A at 25°C 
TPU1170A = Material(rho = 1080, mu = .48, E = lambda freq: 1.4e7*(1+1j*.05))

# Polymere KKP07
KKP07 = Material(rho = 1100, mu = .45, E = lambda freq: 2.68e7*(1+1j*.55))
