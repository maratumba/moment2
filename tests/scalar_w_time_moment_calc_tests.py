# -*- coding: utf-8 -*-
import unittest
from moments.calculators import ScalarMomentCalc


class Scalar1DFunctionWithTime(unittest.TestCase):

    def setUp(self):
        # f(x, t) = x*t
        self.f = lambda x, t: x[0]*t
        interval = [[0, 1]]
        time_interval = [0, 1]
        dx = 0.001
        dt = 0.01
        whole = lambda x: True
        self.calc = ScalarMomentCalc(self.f, interval, [dx], whole,
                                     time_interval, dt)

    def test_moment0(self):
        # (x^2/2)*(t^2/2) at 1 = 1/4
        self.assertAlmostEqual(self.calc.moment0(),
                               1/4, 2)

    def test_moment1(self):
        # (x^3/3)*(t^2/2) =  1/6
        self.assertAlmostEqual(self.calc.moment([0], 0, m=1, ks=[0]),
                               1/6, 2)

    def test_cog(self):
        # (1/6)/(1/4)  = 2/3
        self.assertAlmostEqual(self.calc.center_of_gravity()[0],
                               2/3, 2)

    def test_moment2(self):
        # (x^4/4)*(t^2/2) = 1/8
        self.assertAlmostEqual(self.calc.moment([0], 0, m=2,
                                                ks=[0, 0]),
                               1/8, 2)

    def test_moment2_w_all(self):
        # (x^4/4)*(t^2/2) = 1/8
        self.assertAlmostEqual(self.calc.moment_all([0], 0, m=2)[0][0],
                               1/8, 2)

    def test_time_moment(self):
        # (x^2/2)*(t^3/3) = 1/6
        self.assertAlmostEqual(self.calc.moment([0], 0, n=1),
                               1/6, 2)

    def test_time_moment2(self):
        # (x^2/2)*(t^4/4) = 1/8
        self.assertAlmostEqual(self.calc.moment([0], 0, n=2),
                               1/8, 2)

    def test_time_and_spatial_moment(self):
        # (x^3/3)*(t^3/3) = 1/9
        self.assertAlmostEqual(self.calc.moment_all([0], 0, m=1, n=1)[0],
                               1/9, 2)
