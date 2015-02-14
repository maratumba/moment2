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


def plane(x, y):
    """Returns a point on a plane

    plane(x, y) = start + x*s + y*t
    """
    start = np.array([1, 1, 1])
    s = np.array([1, 0, 0])
    t = np.array([0, 0, 1])
    return start + x*s + y*t


def arrival_to(x, y):
    """Calculates the arrival time to a point
    propagation only goes in `t` direction
    """
    v = 1
    return y/v


def gaussian2d(x, y):
    center = np.array([2.5, 4])
    dist = np.linalg.norm(np.array([x, y]) - center)
    std = 2
    amp = 1
    return amp*math.exp(-1*(dist)**2/(2*std**2))


def trapezoid(t, tw=0):
    """
     __
    /  \
    """
    ts = 0.2  # rising time
    tc = 1    # slip time
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
points = np.array([(x, y) for x in np.linspace(0, 5, 50)
                   for y in np.linspace(0, 8, 80)])


times = np.linspace(0, 10, 100)
arg_len = len(points)*len(times)


def get_args():
    points3d = np.zeros((arg_len, 3))
    times3d = np.zeros(arg_len)
    values = np.zeros((arg_len, 3, 3))
    index = 0
    for x, y in points:
        for time in times:
            points3d[index] = plane(x, y)
            times3d[index] = time
            t_arr = arrival_to(x, y)
            amp = gaussian2d(x, y)
            values[index] = amp*trapezoid(time, t_arr)*M
            index = index + 1
    return points3d, times3d, values


points3d, times3d, values = get_args()
dxs = [0.1, 0.1, 0.1]
dt = 0.1


calc = TensorMomentCalc(values, points3d, times3d,
                        dxs, dt)
print("M0:", calc.tensor_moment0())
center = calc.center_of_gravity()
print("center of gravity: ", center)
m2 = calc.moment_all(center, 0, m=2)
print("M2:", m2)
eig, vcs = np.linalg.eigh(m2)
print("eigenvalues: ", eig)
print("eigenvectors: ", vcs)
tc = calc.moment_all(center, 0, n=1)/calc.moment0()
print("Centeroid time:", tc)
directivity = calc.moment_all(center, tc, m=1, n=1)/calc.moment0()
print("Directivity:", directivity)
