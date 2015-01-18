# -*- coding: utf-8 -*-
import unittest
from moments.calculators import ScalarMomentCalc


class Scalar1DFunction(unittest.TestCase):

    def setUp(self):
        # f(x) = x
        self.f = lambda x, t: x[0]
        interval = [[0, 1]]
        dx = 0.0001
        whole = lambda x: True
        self.calc = ScalarMomentCalc(self.f, interval, [dx], whole)

    def test_moment0(self):
        # x^2/2 at 1 = 1/2
        self.assertAlmostEqual(self.calc.moment0(),
                               0.5, 2)

    def test_moment0_recalc(self):
        self.assertAlmostEqual(self.calc.moment0(),
                               0.5, 2)
        self.assertAlmostEqual(self.calc.moment0(),
                               0.5, 2)

    def test_moment1(self):
        # x^3/3 =  1/3
        self.assertAlmostEqual(self.calc.moment([0], 0, m=1, ks=[0]),
                               1/3, 3)

    def test_cog(self):
        # (1/3)/(1/2)  = 2/3
        self.assertAlmostEqual(self.calc.center_of_gravity()[0],
                               2/3, 3)

    def test_moment2(self):
        # x^4/4 = 1/4
        self.assertAlmostEqual(self.calc.moment([0], 0, m=2,
                                                ks=[0, 0]),
                               1/4, 3)

    def test_moment2_w_all(self):
        # x^4/4 = 1/4
        self.assertAlmostEqual(self.calc.moment_all([0], 0, m=2)[0][0],
                               1/4, 3)
