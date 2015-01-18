# -*- coding: utf-8 -*-
import math
import numpy as np

from moments.calculators import MomentCalc


def distance_to_line(p0, v, point):
    '''http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line'''
    return np.linalg.norm((p0-point)-np.dot((p0-point), v)*v)


def linesource(p, t):
    amp = 1  # amplitude value

    # line(t) = [p0] + t*v*[a]
    p0 = np.array([0, 0])
    v = 1
    alpha = math.pi/4
    a = np.array([math.cos(alpha), math.sin(alpha)])

    t_arr = np.dot(a, p-p0)/v  # Arrival time
    sigma = 0.01
    half_thickness = 0.1  # half thickness of the line
    t_peak = t_arr + 2*sigma

    if t < t_arr or t > t_arr + 4*sigma:
        return 0

    if distance_to_line(p0, a, p) <= half_thickness:
        return amp*math.exp(-1*(t-t_peak)**2/(2*sigma**2))

    return 0


def linesource_optimized(p, t):
    """Optimized version of linesource"""
    amp = 1

    # line(t) = [p0] + t*v*[a]
    p0 = [0, 0]
    v = 1
    alpha = math.pi/3
    a = [math.cos(alpha), math.sin(alpha)]

    z = [p[0]-p0[0], p[1]-p0[1]]
    t_arr = (a[0]*z[0]+a[1]*z[1])/v
    sigma = 0.01
    half_thickness = 0.1
    t_peak = t_arr + 2*sigma

    if t < t_arr or t > t_arr + 4*sigma:
        return 0
    y = ((-z[0]*a[0]-z[1]*a[1]))
    y2 = [y*a[0], y*a[1]]
    y3 = [-z[0]-y2[0], -z[1]-y2[1]]
    leny = math.sqrt(y3[0]**2+y3[1]**2)
    if leny <= half_thickness:
        return amp*math.exp(-1*(t-t_peak)**2/(2*sigma**2))
    return 0


def get_linesource(amp=1, x0=0, y0=0, v=1,
                   alpha=math.pi/2, sigma=0.01, half_thickness=0.1):
    def specialized_linesource(p, t):
        # line(t) = [p0] + t*v*[a]
        p0 = [x0, y0]
        a = [math.cos(alpha), math.sin(alpha)]

        z = [p[0]-p0[0], p[1]-p0[1]]
        t_arr = (a[0]*z[0] + a[1]*z[1]) / v
        t_peak = t_arr + 2*sigma

        if t < t_arr or t > t_arr + 4*sigma:
            return 0
        y = ((-z[0]*a[0]-z[1]*a[1]))
        y2 = [y*a[0], y*a[1]]
        y3 = [-z[0]-y2[0], -z[1]-y2[1]]
        leny = math.sqrt(y3[0]**2+y3[1]**2)
        if leny <= half_thickness:
            return amp*math.exp(-1*(t-t_peak)**2/(2*sigma**2))
        return 0
    return specialized_linesource


limits = [[-1, 1], [-1, 1], [0, 1]]
whole = lambda p: True
dx = 0.01
dt = 0.01

import time as tm

calc = MomentCalc(linesource_optimized, limits, whole, [dx, dx, dt])
# calc.animate_function()
a = tm.time()
print("M0:", calc.moment0())
print("Center of gravity:", calc.center_of_gravity())
print("Directivity:",
      calc.moment_all([0, 0], 0, m=1, n=1)/calc.moment0())
print("completed by", tm.time()-a)
