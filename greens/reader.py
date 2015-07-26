# -*- coding: utf-8 -*-

import os

from greens.utils import g_to_h
from greens.utils import derivative
from greens.utils import parse_greens_name
from greens.utils import get_greens_filename
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


class ReaderNew(object):

    def __init__(self, folder, station_name):
        self.folder = os.path.join(folder,
                                   station_name+"_greens")
        self.name = station_name

    def get(self, greens_name):
        """Get green's function from filename

        Args:
            greens_name (str): dt^n (G|H)_nk,lij

        Returns:
            obspy.core.stream.Stream: green's function
        """
        filename = get_greens_filename(greens_name,
                                       self.name)+"i"
        file_data = parse_greens_name(greens_name)
        filepath = os.path.join(self.folder, filename)
        st = read(filepath)
        if file_data['is_h']:
            st = g_to_h()

        return derivative(st, file_data['n'])
