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
#mpl.style.use('classic')
from mpl_toolkits import mplot3d

plt.rc('font', family='sans-serif',size=8)
plt.rc('axes', labelsize=9)
plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rc('legend',fontsize=8)

#%% 3D opti
def opti_Lc3(geo,f, hc, eta, dd1=0.08, diss=0):

    Hd = 0.06
    dd = dd1
    mu = 0.4
    Hc=hc
    # eta=0.01
    eta=eta
    ind=list(range(len(geo)))
    ind.pop(1)
    Lc=geo[ind]
    Hp=geo[1]
    
    if diss==0: #else just use the frequencies provided
        f = np.linspace(100, 1000, num=91)
    
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
    
    cavity = Cavity3D(height = Hc, r=r, s=s, t=t, zetarst=0.01, medium=fluid1)
    lining = [SinglePlateResonator3D(length=Lc[i], depth=dd, j=j, k=k, l=l, n=n, t=t, plate=plate1, cavity=cavity) for i in range(len(ind))]
    ductelement = [DuctElement3D(lining=lining[i], medium=fluid1, M=0.0) for i in range(len(ind))]
    duct = Duct3D(freq=f, height_d=Hd, elements=list(ductelement))
    return duct.scatteringcoefficients('forward')


#%% Plot
f=np.linspace(250, 1000, num=31)
colors=['black', 'red', 'blue', 'green']
#names=['Hc_1c_LC005', 'Hc_1c_LC01', 'Hc_1c_LC01_modes', 'Hc_1c_LC015', 'Hc_1c_LC02', 'Hc_1c_LC03', 'Hc_2c_LC0025','Hc_2c_LC005', 'Hc_2c_LC01', 'Hc_2c_LC015',  'Hc_3c_LC0033',  'Hc_3c_LC005']
# names=['Hc_1c_LC005', 'Hc_1c_LC01',  'Hc_1c_LC015', 'Hc_1c_LC02', 'Hc_1c_LC03', 'Hc_2c_LC0025','Hc_2c_LC005', 'Hc_2c_LC01', 'Hc_2c_LC015',  'Hc_3c_LC0033',  'Hc_3c_LC005']
names=['Hc_1c_LC015', 'Hc_2c_LC01',  'Hc_3c_LC005']
names=['Hc_1c_LC005', 'Hc_1c_LC01',  'Hc_1c_LC015', 'Hc_1c_LC02', 'Hc_1c_LC03', 'Hc_2c_LC0025','Hc_2c_LC005', 'Hc_2c_LC01', 'Hc_2c_LC015',  'Hc_3c_LC0033',  'Hc_3c_LC005',  'Hc_3c_LC01',  'Hc_5c_LC005']
names=[ 'Hc_1c_LC01_pm10_Alu_eta1']
# names=[ 'Hc_1c_LC01_pm10_Alu_eta01']
names=[ 'Hc_1c_LC005_pm10_Alu_eta01','Hc_1c_LC01_pm10_Alu_eta01', 'Hc_1c_LC01_pm10_Alu_eta1','Hc_1c_LC01_pm10_Alu_eta10', 'Hc_1c_LC015_pm10_Alu_eta01', 'Hc_1c_LC015_pm10_Alu_eta1', 'Hc_2c_LC005_pm10_Alu_eta1', 'Hc_2c_LC01_pm10_Alu_eta1', 'Hc_2c_LC015_pm10_Alu_eta1', 'Hc_2c_LC0.2_pm10_Alu_eta1', 'Hc_3c_LC0.1_pm10_Alu_eta1', 'Hc_3c_LC0.25_pm10_Alu_eta1', 'Hc_3c_LC0.25_pm10_Alu_eta10', 'Hc_4c_LC0.1_pm10_Alu_eta1', 'Hc_4c_LC0.2_pm10_Alu_eta1', 'Hc_5c_LC0.1_pm10_Alu_eta1']

#pareto_data = np.load('./data/pareto_anwendung_data_3D_Hc_1c_LC015.npy')
marker=['o','s', 'p','P']
nplot=2
mplot=2
# colors=['k', 'r', 'b', 'g', 'k', 'r', 'b', 'b', 'b', 'b', 'b']
colors=['darkred','red', 'red', 'lightcoral', 'salmon', 'mistyrose','midnightblue', 'blue', 'royalblue', 'cornflowerblue', 'darkgreen', 'green','lime','darkorange','wheat','k','k','k']
# colors=['darkred','midnightblue',  'darkgreen']
# colors=['darkred','red',  'lightcoral', 'mistyrose']

# f_res = np.array([2, 10, 20])
# f_res=5
Hc=np.linspace(0.01, 0.15, num=15)
f_aus=5# [200 300 400-2 500 600-4 700 800 900]
f_opti=np.linspace(200, 900, num=8)
hc=8
f = np.linspace(250, 1000, num=76)
f = np.linspace(100, 1000, num=91)
for i, n in enumerate(names):
    # for  i, f_aus in enumerate(np.array([1, 3, 5, 7])):

        pareto_data = np.load('./data/pareto_anwendung_data_3D_'+n+'.npy')
        pareto_bounds = np.load('./data/pareto_anwendung_bounds_3D_'+n+'.npy')
        plt.figure(1, figsize=(20/2.54, 15/2.54))
        
        plt.subplot(mplot,nplot,1)
        plt.plot(Hc, -pareto_data[f_aus, 0, :], color=colors[i])
        # plt.vlines(Hc[hc], 0, 4, color='g', linestyle='dashed')
        plt.subplot(mplot,nplot,2)
        # plt.title('Resonator width = 8cm, ' + str(f_opti[f_aus]) + ' Hz')
        plt.plot(Hc, -pareto_data[f_aus, 0, :]/pareto_data[f_aus, 1, :], color=colors[i])  
        # plt.vlines(Hc[hc], 5, 30, color='g', linestyle='dashed')
        
        if 'a01' in n:
            eta=0.001
        if 'a1' in n:
            eta=0.01
        if 'a10' in n:
            eta=0.1
            
        TL=np.zeros([len(f), 4])
        tra1, ref1, dis1 = opti_Lc3(pareto_data[f_aus, 1:, hc], hc=Hc[hc], eta=eta, f='dummy') #4:600 Hz, - ,  10: Hc=0.06, 
        TL[:,0]=-10*np.log10(tra1)
        # TL[:,1]=tra1
        # TL[:,2]=ref1
        # TL[:,3]=dis1
        
        tra=sum(tra1[np.where(f_opti[f_aus]==f)[0][0]-7:np.where(f_opti[f_aus]==f)[0][0]+8]) #f_opti +- 70 Hz
        opti=-10*np.log10(tra/15) # 15 frequenzstützstellen
        plt.figure(2, figsize=(20/2.54, 15/2.54))
        plt.subplot(1,2,1)
        # plt.subplot(mplot,nplot,3)
        plt.plot(f, TL[:,0], '-o', color=colors[i])
        plt.fill_between([f_opti[f_aus]-70, f_opti[f_aus]+70], [opti, opti], alpha=0.4, color=colors[i])
        plt.ylim(0,12)
        plt.subplot(1,2,2)
        plt.plot(f, dis1, color=colors[i])
 
        
        plt.figure(1, figsize=(20/2.54, 15/2.54))
        plt.subplot(mplot,nplot,3)
        plt.plot(Hc, pareto_data[f_aus, 2, :], color=colors[i])
        plt.xlabel('cavity height in m')
        plt.ylabel('hp in m')
        plt.ylim(0.001, 0.01)
        
        plt.subplot(mplot,nplot,4)
        plt.plot(Hc, pareto_data[f_aus, 1, :], color=colors[i])
        plt.xlabel('cavity height in m')
        plt.ylabel('Lc in m')
        # plt.ylim(0.9*10**9, 3.1*10**9)


        # print(opti)
        
plt.suptitle('Optimized for ' + str(f_opti[f_aus]) + ' Hz')
# plt.suptitle('5 chamber, $L_C$=5cm')
plt.subplot(mplot,nplot,1)

plt.xlabel('cavity height in m')
plt.ylabel('TL in dB')

plt.subplot(mplot,nplot,2)
plt.xlabel('cavity height in m')
plt.ylabel('TL in dB/m')
lines = [plt.Line2D([0], [0], color=c, linewidth=3) for c in colors]
# labels = ['$L_C$=5cm','$L_C$=10cm', 'modes $L_C$=10cm','$L_C$=15cm','$L_C$=20cm','$L_C$=30cm','2$L_C$=2.5cm','2$L_C$=5cm','2$L_C$=10cm','2$L_C$=15cm','3$L_C$=3.3cm','3$L_C$=5cm']
labels = names
plt.legend(lines, labels,loc='lower right', fontsize=8, ncol=1, bbox_to_anchor=(1.5, -0.4))
plt.subplots_adjust(left=0.07, right=0.8, bottom=0.08,top=0.95,wspace=0.18,hspace=0.18)


plt.figure(2, figsize=(20/2.54, 15/2.54))
plt.subplot(1,2,1)
plt.legend(lines, labels,loc='upper right', fontsize=8, ncol=1)
plt.ylabel('TL in dB')
plt.xlabel('frequency in Hz')
plt.suptitle('Resonator width = 8cm, ' + str(f_opti[f_aus]) + ' Hz, $H_C$=' + str(np.round(Hc[hc], 3)) + ' m')
plt.subplot(1,2,2)
# plt.legend(lines, labels,loc='upper right', fontsize=8, ncol=1)
plt.ylabel('diss')
plt.xlabel('frequency in Hz')
#plt.savefig('TL_4_selected_LC05_5c_freqs.pdf')


      
    


