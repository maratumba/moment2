# -*- coding: utf-8 -*-
import itertools
import numpy as np

from moments.utils import grid
from moments.utils import crange
from moments.utils import animate_1d_function
from moments.utils import animate_2d_function


class DiscreteScalarMomentCalc(object):
    def __init__(self, values, points, times, dxs, dt):
        """Points in time and space
        """
        self.dimension = len(dxs)
        self.values = values
        self.points = points
        self.times = times
        self.dxs = dxs

        # Multiply all dx values to find dV
        self.dv = 1
        for dx in self.dxs:
            self.dv = self.dv*dx

        self.dt = dt
        self.m0 = None

    def moment(self, q, tau, m=0, n=0, ks=[]):
        """Calculates and returns moment of the function.

        Args:
            q (list): point to calculate moment on ([x, y, z]).
            tau (float): time to calculate moment on.
            m (int): spatial degree of moment.
            n (int): temporal degree of moment.
            ks (list): spatial components used to calculate moment.
              Values for k are 0, 1 and 2 for x, y and z, respectively.
              For example, to calculate
              moment on x and z ks should be ks = [0, 2]
        Returns:
            float: Calculated moment value
        """
        vs = self.values*self.dv*self.dt
        for k in ks:
            vs *= (self.points[:, k] - q[k])
        for j in range(n):
            vs *= (self.times - tau)
        return vs.sum()

    def moment0(self, recalc=False):
        """Shortcut to calculate zero order moment

        Args:
            recalc (bool, optional): calculate the value even if it is
              calcuted before. Otherwise, it will return the cached value.

        Returns:
            float: value of zero order moment
        """
        if self.m0 and not recalc:
            return self.m0
        self.m0 = self.moment(np.zeros(self.dimension), 0, m=0, n=0)
        return self.m0

    def moment_all(self, q, tau, m=0, n=0):
        """Calculates moment for all spatial component combination.

        Works for m <= 2. For m = 1 it calculates moment for all
        components and returns a vector. For m = 2, it calculates
        moment for all two component combinations and returns a
        matrix.

        Args:
            q (list): point to calculate moment on ([x, y, z])
            tau (float): time to calculate moment on
            m (int): spatial degree of moment. Should be less than 3.
            n (int): temporal degree of moment

        Returns:
            numpy.ndarray: moment values. For m = 1 it is a vector, for
                           m =  2 it is a matrix.

        """
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
        """Calculates and returns center of gravity.

        It is calculated by dividing first order moment at origin by
        zero order moment.
        """
        self.m0 = self.moment0()
        m1 = self.moment_all(np.zeros(self.dimension), 0, m=1, n=0)
        return m1/self.m0


class ScalarMomentCalc(DiscreteScalarMomentCalc):
    """Moment Calculator for scalar valued functions

    Args:
        function (function): function to calculate moment for
        space_intervals (list): list of intervals for space components
        dxs (list): list of sample rates for space components
        domain (function): function for determining if a point is in
                           the domain or not
        time_interval (list, optional): interval for time
        dt (float, optional): sample rate for time component

    Examples:

        Let's say we have a bell shaped function

        .. math :: f(x) = 1 - x^2 - y^2

        We can define this function as below. For compability reasons we need
        to have unused time argument for this function.

        >>> bell = lambda x, t: 1 - x[0]**2 - x[1]**2

        Let's say our domain is the rectangular area between -1 and 1 for x
        and y. We can define these intervals as:

        >>> x_interval = [-1, 1]
        >>> y_interval = [-1, 1]
        >>> intervals = [x_interval, y_interval]

        We can define sample rate for x and y 0.01 by

        >>> dx = dy = 0.01

        Our function can get negative results for selected intervals. By
        introducing `positive` function we can restrict our domain for points
        which yield positive values.

        >>> positive = lambda x: bell(x, 0) >= 0

        Calculator object can be defined as:

        >>> calc = ScalarMomentCalc(bell, intervals, [dx, dy], positive)

        Then zero order moment can be calculated using:

        >>> print(calc.moment0())
        1.5707932

        First order moment at origin for `x` component using:

        >>> print(calc.moment([0, 0], 0, m=1, ks=[0]))
        -2.77555756156e-17

        First order moment at origin for `y` component using:

        >>> print(calc.moment([0, 0], 0, m=1, ks=[1]))
        6.50521303491e-19

        `moment_all` function can be used for calculating both of them.

        >>> print(calc.moment_all([0, 0], 0, m=1))
        [ -2.77555756e-17   6.50521303e-19]

        Center of gravity can be calculated using `center_of_gravity`
        function.

        >>> center = calc.center_of_gravity()
        >>> print(center)
        [ -1.76697834e-17   4.14135549e-19]

        We can find second order moments at center of gravity for all
        component combinations using:

        >>> print(calc.moment_all(center, 0, m=2))
        [[  2.61797818e-01  -1.84314369e-18]
         [ -1.84314369e-18   2.61797818e-01]]

        This matrix has values for xx, xy, yx and yy components.

    """
    def __init__(self,  function, space_intervals, dxs, domain,
                 time_interval=[1, 1], dt=1):
        self.points = grid(space_intervals, dxs, domain)
        self.times = crange(time_interval[0], time_interval[1], dt)
        self.dimension = len(dxs)
        self.function = function
        values, args = self._discretize()
        super(ScalarMomentCalc, self).__init__(values,
                                               args['points'],
                                               args['times'],
                                               dxs, dt)

    def _discretize(self):
        values = []
        args = np.array([(p, t) for t in self.times
                         for p in self.points],
                        dtype=[('points', np.float64,
                                (self.dimension,)),
                               ('times', np.float64)])
        for point, time in args:
            values.append(self.function(point, time))
        values = np.array(values)
        return (values, args)

    def animate_function(self):
        """Animates 1D and 2D time based functions."""
        if self.dimension == 1:
            animate_1d_function(self.function, self.points, self.times)
        elif self.dimension == 2:
            animate_2d_function(self.function, self.points, self.times)
        else:
            print("cannot animate")
