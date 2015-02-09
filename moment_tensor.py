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
points = np.array([[0, 0], [1, 1], [2, 2]])
times = np.ones(len(points))
dxs = [1, 1]
dt = 1
values = np.zeros((len(points), 3, 3))
for i, point in enumerate(points):
    values[i] = M

calc = TensorMomentCalc(values, points, times, dxs, dt)
print("M0:", calc.tensor_moment0())
center = calc.center_of_gravity()
print("center of gravity: ", center)
m2 = calc.moment_all(center, 1, m=2)
eig, vcs = np.linalg.eig(m2)
print("eigenvectors: ", vcs)
