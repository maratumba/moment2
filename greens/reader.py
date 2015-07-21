# -*- coding: utf-8 -*-

import os

from greens.utils import g_to_h
from greens.utils import derivative

from obspy.core import read


class Reader(object):

    def __init__(self, folder, station_name):
        self.folder = folder
        self.name = station_name

    def _filepath(self, n, k, l, spatial):
        c = 'xyz'
        filename = self.name + "_" + c[k] + c[l] + spatial + '.' + n + 'i'
        return os.path.join(self.folder, filename)

    def get(self, n, k, l, spatial='', temporal=0):
        """
        d^{temporal}Gnk,l{spatial}
        --
        dt^{temporal}

        n -> r, t, z
        k,l -> x, y, z
        """
        st = read(self._filepath(n, k, l, spatial))
        return derivative(st, temporal)

    def get_h(self, n, k, l, spatial='', temporal=0):
        g = self.get(n, k, l, spatial, temporal)
        return g_to_h(g)


class ReaderForce(object):

    def __init__(self, folder, station_name):
        self.folder = folder
        self.name = station_name

    def _filepath(self, n, k, spatial):
        c = 'xyz'
        filename = self.name + "_" + c[k] + spatial + '.' + n + 'i'
        return os.path.join(self.folder, filename)

    def get(self, n, k, spatial='', temporal=0):
        """
        d^{temporal}Gnk,{spatial}
        --
        dt^{temporal}

        n -> r, t, z
        k,l -> x, y, z
        """
        st = read(self._filepath(n, k, spatial))
        return derivative(st, temporal)

    def get_h(self, n, k, spatial='', temporal=0):
        g = self.get(n, k, spatial, temporal)
        return g_to_h(g)
