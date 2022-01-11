# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) 2021,
#------------------------------------------------------------------------------
"""Implements testing of classes
"""

import unittest
import numpy as np

from acouplasi import DuctElement , Duct ,Temperature, Fluid, Material, SinglePlateResonator,\
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
#Polyurethane TPU1195A
TPU1195A = Material(rho = 1150, mu = .48, E = lambda freq, temp: 3.1e8*.97**(temp.C-7*np.log10(freq))+1j*4.7e7*.96**(temp.C-7*np.log10(freq)))

plate1 = SimplePlate(hp=0.0003, material=TPU1195A, temperature=temp1)

cavity1 = Cavity2D(height=1, r=r, t=t, medium=fluid1)

lining1 = SinglePlateResonator(length=5, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)
lining2 = SimpleTwoSidedPlateResonator(length=5, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)
lining3 = SinglePlateResonator(length=10, depth=1, j=j, l=l, t=t, plate=plate1, cavity=cavity1)

ductelement1 = DuctElement(lining=lining1, medium=fluid1, M=0)
ductelement2 = DuctElement(lining=lining2, medium=fluid1, M=0)
ductelement3 = DuctElement(lining=lining3, medium=fluid1, M=0)

duct1 = Duct(freq=f, height_d=1, elements=[ductelement1])
duct2 = Duct(freq=f, height_d=1, elements=[ductelement1, ductelement2])
duct3 = Duct(freq=f, height_d=1, elements=[ductelement3])


class Test_Silencer2d(unittest.TestCase):
    """Test that ensures that the calculation routines of 
    a 2d plate silence did not change
    """

    def test_TL(self):
        """ test that Transmission loss did not changed"""
        # calculate TL
        TL1 = duct1.tl()
        TL2 = duct2.tl()
        TL3 = duct3.tl()
        self.assertEqual(TL1[3],2.1243861076529913)
        self.assertEqual(TL2[3],2.1243861076529913)
        self.assertEqual(TL3[3],4.853366027037131)
        
        
    def test_transmission_and_reflection(self):
        """ test that factors did not change"""
        t1,r1,d1 = duct1.coefficients()
        t2,r2,d2 = duct2.coefficients()
        t3,r3,d3 = duct3.coefficients()
        self.assertEqual(t1[3],0.613142456124456)
        self.assertEqual(r2[3],0.04397566673739521)
        self.assertEqual(d3[3],0.4922189731123288)


if __name__ == '__main__':
    unittest.main()
