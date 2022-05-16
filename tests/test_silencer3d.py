# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) 2021,
#------------------------------------------------------------------------------
"""Implements testing of classes
"""

import unittest
import numpy as np

from acouplasi import  DuctElement3D , Duct3D ,Temperature, Fluid, Material, TPU1195A, \
SinglePlateResonator3D, SimpleTwoSidedPlateResonator3D, \
SimplePlate3D,  Cavity3D, CavityAlt3D

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
material2 = TPU1195A

plate1 = SimplePlate3D(hp=0.0003, material=material1, temperature=temp1)
plate2 = SimplePlate3D(hp=0.0003, material=material2, temperature=temp1)

cavity1 = Cavity3D(height=1, r=r, s=s, t=t, zetarst=0.01, medium=fluid1)
cavity2 = CavityAlt3D(height=0.5, r=r, s=s, medium=fluid1)
cavity3 = Cavity3D(height=1.3, r=r, s=s, t=t, zetarst=0.01, medium=fluid1)

lining1 = SinglePlateResonator3D(length=5, depth=1, j=j, l=l, k=k, n=n, t=t, plate=plate1, cavity=cavity1)
lining2 = SimpleTwoSidedPlateResonator3D(length=6, depth=0.7, j=j, l=l, k=k, n=n, t=t, plate=plate2, cavity=cavity2)
lining3 = SinglePlateResonator3D(length=7.5, depth=1.2, j=j, l=l, k=k, n=n, t=t, plate=plate1, cavity=cavity3)

ductelement1 = DuctElement3D(lining=lining1, medium=fluid1, M=0)
ductelement2 = DuctElement3D(lining=lining2, medium=fluid1, M=0)
ductelement3 = DuctElement3D(lining=lining3, medium=fluid1, M=0)

duct1 = Duct3D(freq=f, height_d=1, elements=[ductelement1])
duct2 = Duct3D(freq=f, height_d=0.7, elements=[ductelement1, ductelement2])
duct3 = Duct3D(freq=f, height_d=1.1, elements=[ductelement3])

Data = np.load('silencer3d.npy', allow_pickle = True)


class Test_Silencer3d(unittest.TestCase):
    """Test that ensures that the calculation routines of 
    a 3d plate silence did not change
    """
        
    def test_transmission_and_reflection(self):
        """ test that factors did not change"""
        
        t1,r1,d1 = duct1.scatteringcoefficients()
        t2,r2,d2 = duct2.scatteringcoefficients()
        t3,r3,d3 = duct3.scatteringcoefficients()
        
        self.assertEqual(t1[4],Data[0][0][4])
        self.assertEqual(r1[4],Data[0][1][4])
        self.assertEqual(d1[4],Data[0][2][4])
        
        self.assertEqual(t2[4],Data[1][0][4])
        self.assertEqual(r2[4],Data[1][1][4])
        self.assertEqual(d2[4],Data[1][2][4])
        
        self.assertEqual(t3[4],Data[2][0][4])
        self.assertEqual(r3[4],Data[2][1][4])
        self.assertEqual(d3[4],Data[2][2][4])


if __name__ == '__main__':
    unittest.main()
