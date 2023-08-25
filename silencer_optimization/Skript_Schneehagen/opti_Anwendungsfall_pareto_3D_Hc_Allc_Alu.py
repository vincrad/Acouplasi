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

# function to optimize
def opti_Lc(geo, *f, diss=0):
	
    #hier kannst du noch parameter über *f übergeben		
    Hd = 0.06
    dd = 0.08
    mu = 0.4
    Hc=f[1]
    eta=0.01
    #eta=eta
    # das müsstest du dann anpassen, fall du zum Beispiel E oder rho auch optimieren willst
    ind=list(range(len(geo)))
    ind.pop(1) # hier wird nur der eintrag für die höhe entfernt und der rest sind dann nur noch die längen die dann den cavities übergeben werden
    Lc=geo[ind]
    Hp=geo[1]
  	
    #der Teil gibt verscheidene Möglichkeiten für die Optimierung der Breitbandigkeit: plusminus 70 Hz oder Prozent ...		
    if diss==0: #else just use the frequencies provided
        f=f[0]
        f = np.linspace(f-0.1*f, f+0.1*f, num=15) #modespm10
        #f = np.linspace(f-0.1*f, f+0.1*f, num=5) #modespm10_5f
        #f = np.linspace(f-70, f+70, num=5) #modes_5f
        #f=np.append(f, 1000.)
    temp1 = Temperature(C=25)
    
    fluid1 = Fluid(temperature=temp1)
    material1= Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*eta))
    
    plate1 = SimplePlate3D(hp = Hp, material = material1, temperature = temp1)

    # mode orders
    # number of modes 3D
    J = np.ceil(np.sqrt(np.max(f)*2*np.max(Lc)**2/np.pi*np.sqrt(plate1.mass()/np.real(plate1.bendingstiffness(np.max(f), temp1)))))
    K = np.ceil(np.sqrt(np.max(f)*2*dd**2/np.pi*np.sqrt(plate1.mass()/np.real(plate1.bendingstiffness(np.max(f), temp1)))))
    R = np.ceil(2*np.max(f)*np.max(Lc)/np.real(fluid1.c))
    S = np.ceil(2*np.max(f)*dd/np.real(fluid1.c))
    T = np.ceil(2*np.max(f)*Hc/np.real(fluid1.c))
    
    # plate modes
    j = np.arange(1, J+1, 1, dtype='int')
    k = np.arange(1, K+1, 1, dtype='int')
    l = np.arange(1, J+1, 1, dtype='int')
    n = np.arange(1, K+1, 1, dtype='int')
    
    # cavity modes
    r = np.arange(0, R+J, 1, dtype='int')
    s = np.arange(0, S+K, 1, dtype='int')
    t = np.arange(0, 5*T, 1, dtype='int')
    
    # jede kammer hat das gleiche Plattenmaterial, und in 'ind' stehen dann nur noch die anzahl der kammern, die dann hier durch iteriert werden
    cavity = Cavity3D(height = Hc, r=r, s=s, t=t, zetarst=0.01, medium=fluid1)
    lining = [SinglePlateResonator3D(length=Lc[i], depth=dd, j=j, k=k, l=l, n=n, t=t, plate=plate1, cavity=cavity) for i in range(len(ind))]
    ductelement = [DuctElement3D(lining=lining[i], medium=fluid1, M=0.0) for i in range(len(ind))]
    duct = Duct3D(freq=f, height_d=Hd, elements=list(ductelement))
    
    #hier wird das optimierungsziel festgelegt. der Wert 'opti' soll minimal werden. In diesem Fall die der 10log der Summe der Transmissiongrade (aus dem festgelegten Frequenzbereich) geteilt durch die Anzahl der Frequenzstützstellen, für die optimiert wurde.
    tra1, ref1, dis1 = duct.scatteringcoefficients('forward')
    tra=sum(tra1)
    opti=10*np.log10(tra/len(f))#+10*np.log10(dis/len(f[:-1])) #white noise with amplitude 1: (tra*1)/(1)

    
    return opti
    
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


