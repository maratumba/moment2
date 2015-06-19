#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os

from greens.reader import Reader

from moments.utils import read_tensor_moment_info_file

from obspy.core.trace import Trace
from obspy.core.stream import Stream

import math
import numpy as np


def calc_station(folder, name, component, m0, moments):
    """Calculate station seismogram for given moments and green's
    functions (and their derivatives)

    Args:
        folder (str): folder that contains green's functions
        name (str): station name
        component (str): component to calculate (r, t, z)
        m0 (numpy.ndarray): normalized moment tensor
        moments (dict): moments info object

    Returns:
        obspy.core.stream.Stream: obspy stream that contains the calculated
                                  seismogram
    """
    r = Reader(folder, name)

    data = None
    for m in range(3):
        for n in range(3):
            if m+n <= 2:
                f = moments[(m, n)]
                # FIXME: divide by 10^15 dyne/cm^2 ?
                co = ((-1)**n)/(math.factorial(m)*math.factorial(n))/1e15
                for k in range(3):
                    for l in range(3):
                        if m == 0:
                            st = r.get_h(component, k, l, '', n)
                            tr = st[0]
                            if data is None:
                                data = Trace(co*f*m0[k][l]*tr.data, tr.stats)
                            else:
                                data.data = data.data + co*f*m0[k][l]*tr.data
                        elif m == 1:
                            c = 'xyz'
                            for i in range(3):
                                st = r.get_h(component, k, l, c[i], n)
                                tr = st[0]
                                data.data = data.data + co*f[i]*m0[k][l]*tr.data
                        else:  # m == 2
                            c = 'xyz'
                            for i in range(3):
                                for j in range(3):
                                    st = r.get_h(component, k, l, c[i]+c[j], n)
                                    tr = st[0]
                                    data.data = data.data + co*f[i][j]*m0[k][l]*tr.data

    stream = Stream(traces=[data])
    return stream


def write_stream_as_sac(stream, folder, filename, verbose=False):
    """Writes stream as a sac file

    Args:
        stream (obspy.core.stream.Stream): stream object
        folder (str): folder to store the file
        filename (str): file name
        verbose (bool): print what you did
    """
    stream.write(os.path.join(folder, filename), 'SAC')
    if verbose:
        print(os.path.join(folder, filename) + " is written.")


def station_path(root_path, station_name):
    """Returns station folder path
    `root_path`/`station_name`_greens

    Args:
        root_path (str): root path for Green's function
        station_name (str): station name

    Returns:
        str: full station path
    """
    return os.path.join(root_path, station_name+'_greens')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate seismograms from moments and Green\'s functions')
    parser.add_argument("momentfile", help="moment file")
    parser.add_argument("greens_path",
                        help="folder that contains green's functions")
    parser.add_argument("station_names", nargs="*",
                        help="station name (assumes all stations if omitted)")
    parser.add_argument("-o", "--output-folder",
                        help="output folder for seismograms (./seismograms)",
                        default="seismograms")
    parser.add_argument('-c', '--components', nargs='*',
                        default=['r', 't', 'z'],
                        help='components to calculate default r, t, z')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # read moment information file
    moments = read_tensor_moment_info_file(args.momentfile)
    m0 = moments['m0']/np.sum(moments['m0'])

    if not args.station_names:  # if no station name is given
        folders = glob.glob(os.path.join(args.greens_path, '*_greens'))
        names = [folder.split('/')[-1].split('_')[0] for folder in folders]
        args.station_names = names

    # calculate seismogram and write it as a sac file
    for station_name in args.station_names:
        for component in args.components:
            st = calc_station(station_path(args.greens_path,
                                           station_name),
                              station_name, component,
                              m0, moments)
            write_stream_as_sac(st, args.output_folder,
                                station_name+'.'+component,
                                args.verbose)
