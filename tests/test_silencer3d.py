# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) 2021,
#------------------------------------------------------------------------------
"""Implements testing of classes
"""

import unittest
import numpy as np

from acouplasi import  DuctElement3D , Duct3D ,Temperature, Fluid, Material,\
SinglePlateResonator3D, SimplePlate3D,  Cavity3D

# frequency
f = np.arange(10, 260, 10)
# temperature
temp1 = Temperature(C=20)
# # number of plate modes
N = 1
# plate modes
j = np.arange(1, N+1, 1)
k = np.arange(1, N+1, 1)
l = np.arange(1, N+1, 1)
n = np.arange(1, N+1, 1)
# cavity modes
r = np.arange(0,10,1)
s = np.arange(0,10,1)
t = np.arange(0,10,1)

fluid1 = Fluid(temperature=temp1)
material1 = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))
plate1 = SimplePlate3D(hp=0.0003, material=material1, temperature=temp1)
cavity1 = Cavity3D(height=1, r=r, s=s, t=t, medium=fluid1)
#cavity1 = CavityAlt3D(height=1, r=r, s=s, medium=fluid1)
lining1 = SinglePlateResonator3D(length=5, depth=1, j=j, l=l, k=k, n=n, t=t, plate=plate1, cavity=cavity1)
#lining1 = SimpleTwoSidedPlateResonator3D(length=5, depth=1, j=j, l=l, k=k, n=n, t=t, plate=plate1, cavity=cavity1)
ductelement1 = DuctElement3D(lining=lining1, medium=fluid1, M=0)
duct1 = Duct3D(freq=f, height_d=1, elements=[ductelement1])


class Test_Silencer3d(unittest.TestCase):
    """Test that ensures that the calculation routines of 
    a 3d plate silence did not change
    """

    def test_TL(self):
        """ test that Transmission loss did not changed"""
        # calculate TL
        TL1 = duct1.tl()
        self.assertEqual(TL1[4],9.267057002970104)
        
    def test_transmission_and_reflection(self):
        """ test that factors did not change"""
        [tra, ref, dis] = duct1.coefficients()
        self.assertEqual(tra[4],0.11838435156577545)
        self.assertEqual(ref[4],0.7199828692333913)
        self.assertEqual(dis[4],0.1616327792008333)



if __name__ == '__main__':
    unittest.main()
