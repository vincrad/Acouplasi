#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jekosch/radmann/schneehagen
optimizers for plate silencer geometries
"""



import time
from scipy.optimize import  shgo ,minimize,basinhopping,differential_evolution ,brute
from acouplasi import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
from matplotlib import cm
import matplotlib as mpl
mpl.style.use('classic')
from mpl_toolkits import mplot3d




def get_dimensions_de():
    result = differential_evolution(func =  opti_Lc,
                                     bounds=bounds, args=(fi, hi), popsize=100) 
    # die args werden zusätzlich zu den bounds der zu optimierendne funktion übergeben
    return result



def get_dimensions_shgo():
    result = shgo(func =  opti_Lc, bounds=bounds,
                   args=(fi, hi), popsize=100)
    return result

