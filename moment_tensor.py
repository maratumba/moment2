# -*- coding: utf-8 -*-

import numpy as np
from moments.calculators import TensorMomentCalc
import math


def generate_moment_tensor(u, v):
    """Generates and returns moment tensor from given u and v vectors
    for isotropic medium.

    Args:
        u (list): slip vector
        v (list): fault plane normal
    """
    mu = 0.5
    lmbda = 0.5

    M = np.zeros((3, 3))

    d = np.eye(3)  # kronecker delta

    C = lambda i, j, k, l: lmbda*d[i, j]*d[k, l]  \
        + mu*(d[i, k]*d[j, l] + d[i, l]*d[j, k])

    # M_kl = u_i v_j C_ijkl
    for k in range(3):
        for l in range(3):
            total = 0
            for i in range(3):
                for j in range(3):
                    total += u[i]*v[j]*C(i, j, k, l)
            M[k, l] = total

    return M

M = generate_moment_tensor((1, 0, 0), (0, 0, 1))
print("M:", M)


def line(x):
    """Returns a point on a 2d line

    line(x) = start + v*x
    """
    start = np.array([0, 0])
    v = np.array([1, 1])
    return start + v*x


def gaussian(x):
    center = 2.5
    std = 2
    amp = 1
    return amp*math.exp(-1*(x-center)**2/(2*std**2))


def trapezoid(t, tw=0):
    """
     __
    /  \
    """
    ts = 0.2 # rising time
    tc = 1   # slip time
    if t < tw:
        return 0
    t = t - tw
    if t < ts:
        return t/ts
    elif t <= ts + tc:
        return 1
    elif t < 2*ts+tc:
        return 1-(t-ts-tc)/ts
    return 0

# points = np.array([line(x) for x in range(5)])
sample_points = np.linspace(0, 5, 100)
points = np.array([line(x) for x in sample_points])
amps = np.array([gaussian(x) for x in sample_points])
times = np.linspace(0, 9, 900)
arguments = np.array([(p, t, amp) for t in times
                      for p, amp in zip(points, amps)],
                     dtype=[('points', np.float64, (2,)),
                            ('times', np.float64),
                            ('amps', np.float64)])

# get amp from gaussian


def arrival_to(point):
    """Calculates the arrival time to a `point`"""
    v = 1
    start = np.array([0, 0])
    distance = np.linalg.norm(start-point)
    return distance/v


values = np.zeros((len(arguments), 3, 3))
for i, arg in enumerate(arguments):
    point, time, amp = arg
    tw = arrival_to(point)
    values[i, :, :] = amp*trapezoid(time, tw)*M
dxs = [0.05, 0.05]
dt = 0.01

calc = TensorMomentCalc(values, arguments['points'], arguments['times'],
                        dxs, dt)
print("M0:", calc.tensor_moment0())
center = calc.center_of_gravity()
print("center of gravity: ", center)
m2 = calc.moment_all(center, 0, m=2)
eig, vcs = np.linalg.eig(m2)
print("eigenvalues: ", eig)
print("eigenvectors: ", vcs)
tc = calc.moment_all(center, 0, n=1)/calc.moment0()
print("Centeroid time:", tc)
directivity = calc.moment_all(center, tc, m=1, n=1)/calc.moment0()
print("Directivity:", directivity)
