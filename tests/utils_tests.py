# -*- coding: utf-8 -*-
import math

from moments.utils import crange
from moments.utils import grid
from moments.utils import polargrid
from numpy.testing import assert_array_almost_equal


def test_crange():
    values = crange(0, 1, 0.2)
    expected_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
    assert_array_almost_equal(values, expected_values)


def test_1d_grid():
    whole = lambda x: True
    values = grid([[0, 1]], [0.2], whole)
    expected_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
    assert_array_almost_equal(values, expected_values)


def test_2d_grid():
    whole = lambda x: True
    values = grid([[0, 1], [0, 1]], [0.5, 0.5], whole)
    expected_values = [[0, 0], [0, 0.5], [0, 1],
                       [0.5, 0], [0.5, 0.5], [0.5, 1],
                       [1, 0], [1, 0.5], [1, 1]]
    assert_array_almost_equal(values, expected_values)


def test_3d_grid():
    whole = lambda x: True
    values = grid([[0, 1], [0, 1], [0, 1]], [0.5, 0.5, 0.5],
                  whole)
    expected_values = [[0, 0, 0], [0, 0, 0.5], [0, 0, 1],
                       [0, 0.5, 0], [0, 0.5, 0.5], [0, 0.5, 1],
                       [0, 1, 0], [0, 1, 0.5], [0, 1, 1],
                       [0.5, 0, 0], [0.5, 0, 0.5], [0.5, 0, 1],
                       [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 1],
                       [0.5, 1, 0], [0.5, 1, 0.5], [0.5, 1, 1],
                       [1, 0, 0], [1, 0, 0.5], [1, 0, 1],
                       [1, 0.5, 0], [1, 0.5, 0.5], [1, 0.5, 1],
                       [1, 1, 0], [1, 1, 0.5], [1, 1, 1]]
    assert_array_almost_equal(values, expected_values)


def test_polar_grid():
    whole = lambda x: True
    values = polargrid([[0, 1], [0, math.pi/2]], [0.5, math.pi/4],
                       whole)
    sqrt2 = math.sqrt(2)
    expected_values = [[0, 0], [0, 0], [0, 0],
                       [0.5, 0], [sqrt2/4, sqrt2/4], [0, 0.5],
                       [1, 0], [sqrt2/2, sqrt2/2], [0, 1]]
    assert_array_almost_equal(values, expected_values)
