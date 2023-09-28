#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:25:50 2022

@author: schneehagen
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

#%%
Lcmax=0.25
#bounds= [(0.025, 0.085), (0.03, 0.055), (0.003, 0.009), (850, 1150), (1*10**9, 3*10**9)]
bounds= [(0.01, Lcmax), (0.001, 0.01),(0.01, Lcmax),(0.01, Lcmax)] # in diesem array kannst du angeben, was optimiert werden kann, die anzahl der kammern musst aber du entscheiden und das ist kein freier parameter ( könnte man aber noch implementieren)
f=np.linspace(200, 900, num=8) # es wird für 8 Frequenzen optimiert

#Hc=np.linspace(0.01, 0.15, num=29)
Hc=np.linspace(0.01, 0.15, num=15) 
pareto_data=np.zeros([len(f),len(bounds)+1, len(Hc)])
# es wird über die kavitätshöhe und die frquenz iteriert und im anschluss alles abgespeichert
for h, hi in enumerate(Hc):
    for i, fi in enumerate(f):
        
        t0 = time.time()
        
        result = differential_evolution(func =  opti_Lc, bounds=bounds, args=(fi, hi), popsize=100) # die args werden zusätzlich zu den bounds der zu optimierendne funktion übergeben
        
        t1 = time.time()
        print('Computation time: ', np.round(t1-t0,3), ' seconds for ', fi, ' Hz and Hc = ' , hi, ' m')
        print(result.fun, result.x)
        pareto_data[i, 0, h]=result.fun
        pareto_data[i, 1:, h]=result.x

np.save('pareto_anwendung_data_3D_Hc_'+str(len(bounds)-1)+'c_LC'+str(Lcmax)+'_pm10_Alu_eta1', pareto_data)
np.save('pareto_anwendung_bounds_3D_Hc_'+str(len(bounds)-1)+'c_LC'+str(Lcmax)+'_pm10_Alu_eta1', bounds)


