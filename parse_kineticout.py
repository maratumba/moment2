#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
from numpy import cos, sin
import re
import utm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from moments.utils import crange
from moments.utils import write_tensor_moment_info_file
from moments.calculators import TensorMomentCalc


def calc_moment_tensor(strike, dip, rake):
    """Calcute moment tensor from `strike`, `dip` and `rake` according to
    Quantative Seismology, Aki 2002.

    Args:
        strike (float): strike in degrees
        dip (float): dip in degrees
        rake (float): rake in degrees

    Returns:
       numpy.ndarray: moment tensor
    """
    m = np.zeros((3, 3))

    sigma = np.radians(dip)
    phi = np.radians(strike)
    lmbda = np.radians(rake)

    # M_xx
    m[0, 0] = -1*sin(sigma)*cos(lmbda)*sin(2*phi) - sin(2*sigma)*(sin(phi))**2*sin(lmbda)
    # M_xy, M_yx
    m[0, 1] = sin(sigma)*cos(lmbda)*cos(2*phi) + 0.5*sin(2*sigma)*sin(lmbda)*sin(2*phi)
    m[1, 0] = sin(sigma)*cos(lmbda)*cos(2*phi) + 0.5*sin(2*sigma)*sin(lmbda)*sin(2*phi)
    # M_xz, M_zx
    m[0, 2] = -1*cos(sigma)*cos(lmbda)*cos(phi) - cos(2*sigma)*sin(lmbda)*sin(phi)
    m[2, 0] = -1*cos(sigma)*cos(lmbda)*cos(phi) - cos(2*sigma)*sin(lmbda)*sin(phi)
    # M_yy
    m[1, 1] = sin(sigma)*cos(lmbda)*sin(2*phi) - sin(2*sigma)*sin(lmbda)*(cos(phi))**2
    # M_yz, M_zy
    m[1, 2] = -1*cos(sigma)*cos(lmbda)*sin(phi) + cos(2*sigma)*sin(lmbda)*cos(phi)
    m[2, 1] = -1*cos(sigma)*cos(lmbda)*sin(phi) + cos(2*sigma)*sin(lmbda)*cos(phi)
    # M_zz
    m[2, 2] = sin(2*sigma)*sin(lmbda)
    return m


def get_slip_rate_function(rise, fall, rupt_time=0):
    """Returns the slip rate function from given rise and fall times.
    rupt_time delays the function for given amount
    Ji et al. 2003

    Args:
        rise (float): rise time
        fall (float): fall time
        rupt_time (float): rupture start time (shifts function through time)

    Returns:
        function: slip rate function
    """
    ts = rise
    te = fall
    t0 = rupt_time

    def slip_rate_function(t):
        t = t - t0
        c = 1.0/(ts+te)
        if 0 <= t <= ts:
            return c*(1-np.cos((np.pi*t)/ts))
        elif ts < t <= ts + te:
            return c*(1+np.cos((np.pi*(t-ts))/te))
        else:
            return 0
    return slip_rate_function


class Point(object):
    def __init__(self, match, da, mu):
        """Stores parameters for a point and from those it can calculate
        moment_tensor for a given time.

        Args:
            match (match): regular expression match for a file line
            da (float): area (dx*dy)
            mu (float): rigidity (dyne/cm^2)
        """
        self.lon = float(match.group('lon'))
        self.lat = float(match.group('lat'))
        self.depth = float(match.group('depth'))
        self.dist = float(match.group('dist'))
        self.slip = float(match.group('slip'))
        self.rake = float(match.group('rake'))
        self.rupt_time = float(match.group('rupt_time'))
        self.rise = float(match.group('rise'))
        self.fall = float(match.group('fall'))
        self.strike = float(match.group('strike'))
        self.dip = float(match.group('dip'))
        self.da = da
        self.mu = mu
        self.m = None
        self.slip_rate_func = get_slip_rate_function(self.rise, self.fall,
                                                     self.rupt_time)

    def get_moment_tensor(self, recalc=False):
        """Calculates and returns the moment tensor

        Args:
            recalc (bool): recalculate even if it is done before

        Returns:
            numpy.ndarray: moment tensor
        """
        if self.m is not None and not recalc:
            return self.m
        m = calc_moment_tensor(self.strike, self.dip, self.rake)

        # m0 = mu*A*D
        m0 = self.mu*self.da*self.slip
        self.m = m0*m

        return self.m

    def get_moment_tensor_at(self, t):
        """Returns moment tensor at a specific time

        Args:
            t (float): time which moment tensor is calculated

        Returns:
            numpy.ndarray: moment tensor
        """
        return self.slip_rate_func(t)*self.get_moment_tensor()


def readfile(filename, mu):
    """Parses the file and returns `Point` list

    Args:
        filename (str): filename to parse
        mu (float): rigidity value (dyne/cm^2)

    Returns:
       list: Point list
    """
    # parse header using regular expressions
    event_info = re.compile("#evla\s+(?P<evla>[\d\.]+) evlon\s+(?P<evlo>[\d\.]+) hypodepth\s+(?P<evdepth>[\d\.]+)")
    file_header = re.compile('#Total number of fault_segments=\s+(?P<segments>\d+)')
    segment_header = re.compile('#Fault_segment =\s+(?P<seg_id>\d+) nx\(Along\-strike\)=\s+(?P<nx>\d+) Dx=\s+(?P<dx>[\d\.]+)km ny\(downdip\)=\s+(?P<ny>\d+) Dy=\s+(?P<dy>[\d\.]+)km')

    boundary = re.compile('\s+(?P<lon>[\d\.]+)\s+(?P<lat>[\d\.]+)\s+(?P<depth>[\d\.]+)')
    point = re.compile('\s+(?P<lat>[\d\.]+)\s+(?P<lon>[\d\.]+)\s+(?P<depth>[\d\.]+)\s+(?P<dist>[\d\.]+)\s+(?P<slip>[\d\.]+)\s+(?P<rake>[\d\.]+)\s+(?P<rupt_time>[\d\.]+)\s+(?P<rise>[\d\.]+)\s+(?P<fall>[\d\.]+)\s+(?P<strike>[\d\.]+)\s+(?P<dip>[\d\.]+)')
    with open(filename) as f:
        event = event_info.match(f.readline())
        evla = float(event.group('evla'))
        evlo = float(event.group('evlo'))

        ev_utm = utm.from_latlon(evla, evlo)
        ev_x = ev_utm[1]  # northing
        ev_y = ev_utm[0]  # easting

        header = file_header.match(f.readline())
        no_of_segments = int(header.group('segments'))

        total_n_p = 0  # total number of points

        for segment_id in range(no_of_segments):
            # read segment
            segment = segment_header.match(f.readline())
            # print("segment:", segment.groups())
            nx = int(segment.group('nx'))
            ny = int(segment.group('ny'))
            dx = float(segment.group('dx'))
            dy = float(segment.group('dy'))

            n_p = nx*ny

            f.readline()  # Boundary of fault_segment ...
            f.readline()  # Lon. Lat. Depth

            # read boundaries
            for i in range(5):
                boundary.match(f.readline())

            f.readline()  # Lon. Lat. depth dist...

            # read `n_p` points
            points = [Point(point.match(f.readline()), dx*dy, mu)
                      for i in range(n_p)]

            total_n_p += n_p

    # Convert lat, lon to xy coordinate using utm
    xyz = np.zeros((total_n_p, 3))
    for j in range(ny):
        for i in range(nx):
            p = points[i+j*nx]
            t = utm.from_latlon(p.lat, p.lon)
            y = (t[1] - ev_x)/1000  # norting in km
            x = (t[0] - ev_y)/1000  # easting in km
            xyz[i+j*nx] = [x, y, p.depth]
    return (xyz, points, dx, dy)


def plot_points(xyz, points):
    """Plots the points in xyz space"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([p[0] for p in xyz],
               [p[1] for p in xyz],
               [p[2] for p in xyz])
    ax.invert_zaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def generate_time_space_points(points, xyz, dt):
    """Generate time and space points to be used in moment calculation

    Args:
        points (list): Point list obtained from reading kinetic_out file
        xyz (numpy.ndarry): coordinates in xyz space
        dt (float): time interval

    Returns:
        tuple: (values, points, times)
    """
    # find max t
    maxt = max([point.rupt_time + point.rise + point.fall for point in points])
    times = crange(0, maxt, dt)
    values = []
    v_times = []
    v_points = []
    for time in times:
        for point, loc in zip(points, xyz):
            values.append(point.get_moment_tensor_at(time))
            v_times.append(time)
            v_points.append(loc)
    return (np.array(values), np.array(v_points), np.array(v_times))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Parses kinetic_out file and calculates moment values")
    parser.add_argument("filename", help="file to parse")
    parser.add_argument("output", help="file to write moment info")
    parser.add_argument("--plot", help="plot the fault file in xyz space",
                        action="store_true")
    parser.add_argument("--dt", '-d', help="time interval (default is 0.1)",
                        default=0.1, type=float)
    parser.add_argument("--rigidity", '-m',
                        help="rigidity value (default is 3x10^11 dyne/cm^2)",
                        default=3e11, type=float)
    args = parser.parse_args()

    xyz, points, dx, dy = readfile(args.filename, args.rigidity)
    if args.plot:
        plot_points(xyz, points)

    v, p, t = generate_time_space_points(points, xyz, args.dt)

    calc = TensorMomentCalc(v, p, t, [dx, dy, 1], args.dt)
    write_tensor_moment_info_file(calc, args.output)
