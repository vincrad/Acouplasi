# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) 2021,
#------------------------------------------------------------------------------
"""Implements testing of classes
"""

import unittest
import acouplasi as acpl
from traitlets import HasTraits, Int, Float, Bool, Enum

class Test_Instancing(unittest.TestCase):
    """Test that ensures that digest of classes changes correctly on
    changes of CArray and List attributes.
    """

    def test_instancing(self):
        """ test that all classes can be instatiated """
        # iterate over all definitions labels
        for i in dir(acpl):
            with self.subTest(i):
                j = getattr(acpl,i) # class, function or variable
                if isinstance(j,type): # is this a class ?
                    j() # this is an instance of the class

    def test_set_traits(self):
        """ test that important traitlets can be set"""
        # iterate over all definitions labels
        for i in dir(acpl):
            j = getattr(acpl,i) # class, function or variable
            # HasTraits derived class ?
            if isinstance(j,type) \
                and issubclass(j,HasTraits) \
                and ('digest' in j.class_traits().keys()):
                do = j.class_traits()['digest'].depends_on
                if do:
                    obj = j()
                    for k in do:
                        with self.subTest(i+'.'+k):
                            if k in j.class_trait_names():
                                tr = j.class_traits()[k]
                                # handling different Trait types
                                if tr.is_trait_type(Int):
                                    setattr(obj,k,1)
                                elif tr.is_trait_type(Float):
                                    setattr(obj,k,0.1)
                                elif tr.is_trait_type(Bool):
                                    setattr(obj,k,False)

if __name__ == '__main__':
    unittest.main()
