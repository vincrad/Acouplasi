# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) 2021,
#------------------------------------------------------------------------------
"""Implements testing of classes
"""

import unittest
import numpy as np

from acouplasi import DuctElement , Duct ,Temperature, Fluid, \
Material, TPU1195A, \
SinglePlateResonator,\
SimpleTwoSidedPlateResonator , SimplePlate, DoubleLayerPlate,\
DoubleLayerPlate3D, TripleLayerPlate3D, Cavities2D, Cavity2D, CavityAlt2D

# # frequency
f = np.arange(10,260,10)
# # temperature
temp1 = Temperature(C=20)
#number of plate modes
N = 5
# plate modes
j = np.arange(1, N+1, 1)
l = np.arange(1, N+1, 1)
# cavity modes
r = np.arange(0,30,1)
t = np.arange(0,30,1)
fluid1 = Fluid(temperature=temp1)

material1 = Material(rho = 2700, mu = .34, E = lambda freq, temp: 7.21e10*(1+1j*.0001))
material2 = TPU1195A

plate1 = SimplePlate(hp=0.0003, material=material1, temperature=temp1)
plate2 = SimplePlate(hp=0.0003, material=material2, temperature=temp1)

cavity1 = Cavity2D(height=1, r=r, t=t, zetart=0.01, medium=fluid1)
cavity2 = CavityAlt2D(height=0.65, r=r, medium=fluid1)

lining1 = SinglePlateResonator(length=5, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)
lining2 = SimpleTwoSidedPlateResonator(length=4.7, depth=1, j=j, l=l, t=t, plate=plate2, cavity=cavity2)
lining3 = SinglePlateResonator(length=10, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)

ductelement1 = DuctElement(lining=lining1, medium=fluid1, M=0)
ductelement2 = DuctElement(lining=lining2, medium=fluid1, M=0)
ductelement3 = DuctElement(lining=lining3, medium=fluid1, M=0)

duct1 = Duct(freq=f, height_d=1, elements=[ductelement1])
duct2 = Duct(freq=f, height_d=1, elements=[ductelement1, ductelement2])
duct3 = Duct(freq=f, height_d=1, elements=[ductelement3])

Data = np.load('silencer2d.npy', allow_pickle = True)

class Test_Silencer2d(unittest.TestCase):
    """Test that ensures that the calculation routines of 
    a 2d plate silence did not change
    """
        
    def test_scatteringcoefficients(self):
        """ test that coefficients did not change"""
        
        t1,r1,d1 = duct1.scatteringcoefficients()
        t2,r2,d2 = duct2.scatteringcoefficients()
        t3,r3,d3 = duct3.scatteringcoefficients()
        
        self.assertEqual(t1[3],Data[0][0][3])
        self.assertEqual(r1[3],Data[0][1][3])
        self.assertEqual(d1[3],Data[0][2][3])
        
        self.assertEqual(t2[3],Data[1][0][3])
        self.assertEqual(r2[3],Data[1][1][3])
        self.assertEqual(d2[3],Data[1][2][3])
        
        self.assertEqual(t3[3],Data[2][0][3])
        self.assertEqual(r3[3],Data[2][1][3])
        self.assertEqual(d3[3],Data[2][2][3])


if __name__ == '__main__':
    unittest.main()
