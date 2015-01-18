# -*- coding: utf-8 -*-
import itertools
import numpy as np

from moments.utils import grid
from moments.utils import crange
from moments.utils import animate_1d_function
from moments.utils import animate_2d_function


class MomentCalc(object):
    def __init__(self,  function, limits, domain, sample_rates):
        self.function = function
        self.dxs = sample_rates[:-1]

        # multiply dx terms to find dv
        self.dv = 1
        for dx in self.dxs:
            self.dv *= dx

        self.dt = sample_rates[-1]  # last one is for time
        self.points = grid(limits[:-1], sample_rates, domain)
        self.times = crange(limits[-1][0], limits[-1][1], self.dt)
        self.dimension = len(self.dxs)
        self.m0 = None
        self.values = None

    def _calc_values(self, recalc=False):
        if self.values is not None and not recalc:
            return (self.values, self.args)
        self.values = []
        self.args = np.array([(p, t) for t in self.times
                              for p in self.points],
                             dtype=[('points', np.float64, (self.dimension,)),
                                    ('times', np.float64)])
        for point, time in self.args:
            self.values.append(self.function(point, time))
        self.values = np.array(self.values)
        return (self.values, self.args)

    def moment(self, q, tau, m=0, n=0, ks=[]):
        self.values, self.args = self._calc_values()
        ps = self.args['points']
        ts = self.args['times']
        vs = self.values*self.dv*self.dt
        for k in ks:
            vs *= (ps[:, k] - q[k])
        for j in range(n):
            vs *= (ts - tau)
        return vs.sum()

    def moment0(self, recalc=False):
        if self.m0 and not recalc:
            return self.m0
        self.m0 = self.moment(np.zeros(self.dimension), 0, m=0, n=0)
        return self.m0

    def moment_all(self, q, tau, m=0, n=0):
        # if m is equals to 0, there is only one moment
        if m == 0:
            return self.moment(q, tau, m, n)
        # m > 2 case requires unneeded complexity
        if m > 2:
            raise NotImplemented
        indices = range(self.dimension)
        # shape of combinations
        # m = 1 => a dimension length vector
        # m = 2 => a matrix dimension by dimension
        shape = np.ones(m)*self.dimension
        moments = np.zeros(shape)
        for ks in itertools.combinations_with_replacement(indices, m):
            moment = self.moment(q, tau, m, n, ks)
            moments[ks] = moment
            if m == 2:  # Mij = Mji
                moments[tuple(reversed(ks))] = moment
        return moments

    def center_of_gravity(self):
        self.m0 = self.moment0()
        m1 = self.moment_all(np.zeros(self.dimension), 0, m=1, n=0)
        return m1/self.m0

    def animate_function(self):
        if self.dimension == 1:
            animate_1d_function(self.function, self.points, self.times)
        elif self.dimension == 2:
            animate_2d_function(self.function, self.points, self.times)
        else:
            print("cannot animate")
