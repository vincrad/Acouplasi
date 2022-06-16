#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 07:35:22 2020

@author: radmann
"""

import traitlets as tr
import numpy as np
from numpy import log10
from numpy import abs

from .DuctElement import DuctElement, DuctElement3D
 
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
    
    # method calculates the scattering matrix of the overall duct using the wave chain matrices of the certain duct elements
    def scatteringmatrix(self):
        
        # wave chain matrix of the overall duct
        WCM = np.empty((2,2,len(self.freq)), dtype=complex)
        
        for idx1, item1 in enumerate(self.elements):
            
            # calculate scattering matrix of the certain duct element
            sm = item1.scatteringmatrix(self.height_d, self.freq)
            
            # calculate the wave chain matrix of the certain duct element
            wcm = np.empty((2,2,len(self.freq)), dtype=complex)
            
            wcm[0,0,:] = sm[0,1,:]-((sm[0,0,:]*sm[1,1,:])/(sm[1,0,:]))
            wcm[0,1,:] = sm[0,0,:]/sm[1,0,:]
            wcm[1,0,:] = -sm[1,1,:]/sm[1,0,:]
            wcm[1,1,:] = 1/sm[1,0,:]
            
            if idx1 == 0:
                
                WCM = wcm
                
            else:
                
                for idx2, item2 in enumerate(self.freq):
                    
                    WCM[:,:,idx2] = np.dot(WCM[:,:,idx2], wcm[:,:,idx2])
                    
        # scattering matrix of the overall duct
        SM = np.empty((2,2,len(self.freq)), dtype=complex)
        
        SM[0,0,:] = WCM[0,1,:]/WCM[1,1,:]
        SM[0,1,:] = WCM[0,0,:]-((WCM[0,1,:]*WCM[1,0,:])/(WCM[1,1,:]))
        SM[1,0,:] = 1/WCM[1,1,:]
        SM[1,1,:] = -WCM[1,0,:]/WCM[1,1,:]
        
        return SM  
    
    # method calculates the transmission coefficient, reflection coefficient and dissipation coefficient of the overall duct from the overall scattering matrix
    def scatteringcoefficients(self, direction:str):
        
        # mach number
        M = self.elements[0].M
        
        # scattering matrix of the overall duct
        SM = self.scatteringmatrix()
        
        if direction == 'forward':
            
            # transmittance of the overall duct
            tra = np.abs(SM[1,0,:])**2
            
            # reflectance of the overall duct
            ref = ((1-M)**2/(1+M)**2)*np.abs(SM[0,0,:])**2
            
            # dissipation of the overall duct
            dis = 1-tra-ref
            
            return tra, ref, dis
        
        elif direction == 'reverse':
            
            # transmittance of the overall duct
            tra = np.abs(SM[0,1,:])**2
            
            # reflectance of the overall duct
            ref = ((1+M)**2/(1-M)**2)*np.abs(SM[1,1,:])**2
            
            # dissipation of the overall duct
            dis = 1-tra-ref
            
            return tra, ref, dis
        
        else:
            
            print('Incorrect choice. Choose forward or reverse.')
        
#%%
class Duct3D(tr.HasTraits):

    '''
    Combines different DuctElements to build a silencer.
    '''
    
    # frequency
    freq = tr.Instance(np.ndarray)
    
    # duct height
    height_d = tr.Float()

    # list of DuctElements
    elements = tr.List(trait = tr.Instance(DuctElement3D))
    
    # method calculate teh transmission loss of the duct
    def tl(self):
        
        for i in self.elements:
            
            TL = i.tmatrix(self.height_d, self.freq)
            
            return TL
    # method calculates the scattering matrix of the overall duct using the wave chain matrices of the certain duct elements
    def scatteringmatrix(self):
        
        # wave chain matrix of the overall duct
        WCM = np.empty((2,2,len(self.freq)), dtype=complex)
        
        for idx1, item1 in enumerate(self.elements):
            
            # calculate scattering matrix of the certain duct element
            sm = item1.scatteringmatrix(self.height_d, self.freq)
            
            # calculate the wave chain matrix of the certain duct element
            wcm = np.empty((2,2,len(self.freq)), dtype=complex)
            
            wcm[0,0,:] = sm[0,1,:]-((sm[0,0,:]*sm[1,1,:])/(sm[1,0,:]))
            wcm[0,1,:] = sm[0,0,:]/sm[1,0,:]
            wcm[1,0,:] = -sm[1,1,:]/sm[1,0,:]
            wcm[1,1,:] = 1/sm[1,0,:]
            
            if idx1 == 0:
                
                WCM = wcm
                
            else:
                
                for idx2, item2 in enumerate(self.freq):
                    
                    WCM[:,:,idx2] = np.dot(WCM[:,:,idx2], wcm[:,:,idx2])
                    
        # scattering matrix of the overall duct
        SM = np.empty((2,2,len(self.freq)), dtype=complex)
        
        SM[0,0,:] = WCM[0,1,:]/WCM[1,1,:]
        SM[0,1,:] = WCM[0,0,:]-((WCM[0,1,:]*WCM[1,0,:])/(WCM[1,1,:]))
        SM[1,0,:] = 1/WCM[1,1,:]
        SM[1,1,:] = -WCM[1,0,:]/WCM[1,1,:]
        
        return SM  
    
    # method calculates the transmission coefficient, reflection coefficients and dissipation coefficient of the overall duct from the overall scattering matrix
    def scatteringcoefficients(self, direction:str):
        
        # mach number
        M = self.elements[0].M
        
        # scattering matrix of the overall duct
        SM = self.scatteringmatrix()
        
        if direction == 'forward':
            
            # transmittance of the overall duct
            tra = np.abs(SM[1,0,:])**2
            
            # reflectance of the overall duct
            ref = ((1-M)**2/(1+M)**2)*np.abs(SM[0,0,:])**2
            
            # dissipation of the overall duct
            dis = 1-tra-ref
            
            return tra, ref, dis
        
        elif direction == 'reverse':
            
            # transmittance of the overall duct
            tra = np.abs(SM[0,1,:])**2
            
            # reflectance of the overall duct
            ref = ((1+M)**2/(1-M)**2)*np.abs(SM[1,1,:])**2
            
            # dissipation of the overall duct
            dis = 1-tra-ref
            
            return tra, ref, dis
        
        else:
            
            print('Incorrect choice. Choose forward or reverse.')