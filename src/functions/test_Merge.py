#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:21:39 2023

@author: asier
"""
import unittest
import numpy as np
from functions.merge import *
from fmic import FMiC


class TestEuclideanMerge(unittest.TestCase):

    def test_euclidean1(self):
        # For two equal FMiCs and a new example with membership 1 to both
        # means that they should be merged
        fm1 = FMiC([0, 0], None, 0)
        fm2 = FMiC([0, 0], None, 0)
        fms = [fm1, fm2]
        mbs = [1, 1]
        Merger = FuzzyEuclideanMerger(0, 0.8, 5)
        n_fms = Merger.merge(fms, mbs)
        self.assertEqual(len(n_fms), 1)

    def test_euclidean2(self):
        # For two very different FMiCs and a new example with membership 1 to
        # the first and 0 to the second means that they should not be merged
        fm1 = FMiC([0, 0], None, 0)
        fm2 = FMiC([10, 10], None, 0)
        fms = [fm1, fm2]
        mbs = [1, 0]
        Merger = FuzzyEuclideanMerger(0, 0.8, 5)
        n_fms = Merger.merge(fms, mbs)
        self.assertEqual(len(n_fms), 2)


class TestPSProdMerge(unittest.TestCase):

    def test_ps_prod1(self):
        # For two equal FMiCs and a new example with membership 1 to both
        # means that they should be merged
        fm1 = FMiC([0, 0], None, 0)
        fm2 = FMiC([0, 0], None, 0)
        fms = [fm1, fm2]
        mbs = [1, 1]
        Merger = FuzzyPSProdMerger(0, 0.8, 5)
        n_fms = Merger.merge(fms, mbs)
        self.assertEqual(len(n_fms), 1)

    def test_ps_prod2(self):
        # For two very different FMiCs and a new example with membership 1 to
        # the first and 0 to the second means that they should not be merged
        fm1 = FMiC([0, 0], None, 0)
        fm2 = FMiC([10, 10], None, 0)
        fms = [fm1, fm2]
        mbs = [1, 0]
        Merger = FuzzyPSProdMerger(0, 0.8, 5)
        n_fms = Merger.merge(fms, mbs)
        self.assertEqual(len(n_fms), 2)

    def test_ps_prod3(self):
        fm1 = FMiC([10, 10], None, 0)
        fm2 = FMiC([0, 0], None, 0)
        fm3 = FMiC([10, -10], None, 0)
        fm4 = FMiC([-10, -10], None, 0)
        fm5 = FMiC([0, 0], None, 0)
        fm6 = FMiC([-10, 10], None, 0)
        fms = [fm1, fm2, fm3, fm4, fm5, fm6]
        mbs = [0, 1, 0, 0, 1, 0]
        Merger = FuzzyPSProdMerger(0, 0.8, 6)
        n_fms = Merger.merge(fms, mbs)
        self.assertEqual(len(n_fms), 5)

    def test_ps_prod4(self):
        fm1 = FMiC([10, 10], None, 0)
        fm2 = FMiC([0, 0], None, 0)
        fm3 = FMiC([10, -10], None, 0)
        fm4 = FMiC([-10, -10], None, 0)
        fm5 = FMiC([0, 0], None, 0)
        fm6 = FMiC([-10, 10], None, 0)
        fms = [fm1, fm2, fm3, fm4, fm5, fm6]
        mbs = [0, 1, 0, 0, 1, 0]
        Merger = FuzzyPSProdMerger(0, 0.8, 6)
        Merger.similMatrix[:, :, 2] = 1
        Merger.similMatrix[:, :, 1] = 0.5
        Merger.similMatrix[:, :, 2] = np.triu(Merger.similMatrix[:, :, 2], 1)
        Merger.similMatrix[:, :, 1] = np.triu(Merger.similMatrix[:, :, 1], 1)
        n_fms = Merger.merge(fms, mbs)
        self.assertAlmostEqual(fms[1].n, 2)
        self.assertEqual(len(n_fms), 5)
        self.assertEqual(Merger.similMatrix.shape, (6, 6, 3))
        self.assertEqual(Merger.similMatrix[0, 1, 2], 1)
        self.assertEqual(Merger.similMatrix[0, 1, 1], 0.25)
        self.assertEqual(Merger.similMatrix[1, 2, 1], 0.25)
        self.assertEqual(Merger.similMatrix[1, 5, 1], 0)

        # array([[0.  , 0.25, 0.5 , 0.5 , 0.5 , 0.  ],
        #    [0.  , 0.  , 0.25, 0.25, 0.25, 0.  ],
        #    [0.  , 0.  , 0.  , 0.5 , 0.5 , 0.  ],
        #    [0.  , 0.  , 0.  , 0.  , 0.5 , 0.  ],
        #    [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ],
        #    [0.  , 0.  , 0.  , 0.  , 0.  , 0.  ]])


if __name__ == '__main__':
    unittest.main()
