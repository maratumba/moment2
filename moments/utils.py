# -*- coding: utf-8 -*-
import numpy as np
import math

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import juggle_axes
    import matplotlib.animation as animation
except:
    print("no matplotlib")


def crange(start, stop, step):
    """Returns sequence within a closed range

    Args:
        start (float): start number
        stop (float): stop number
        step (float): step between numbers

    Returns:
        numpy.ndarray: A sequence of between start and stop by step

    Examples:
        >>> print(crange(0, 1, 0.1))
        [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
    """
    return np.arange(start, stop+step, step)


def grid(limits, sample_rates, domain):
    """Returns points in a grid for 1D, 2D & 3D.

    Args:
        limits (list): list of limits
        sample_rates (list): list of sample rates
        domain (function): function for determining if point is in
                           the domain or not.

    Returns:
        numpy.ndarray: One dimensional list of all points in the domain.

    Examples:
        Let's say x, y values are between -1 and 1.

        >>> xlimits = [-1, 1]
        >>> ylimits = [-1, 1]

        Sample rates for x and y are 0.5.

        >>> dx = dy = 0.5

        We want points in unit circle (x^2+ y^2 <= 1).

        >>> in_circle = lambda x: x[0]**2 + x[1]**2 <= 1

        This values can be generated using:

        >>> print(grid([xlimits, ylimits], [dx, dy], in_circle))
        [[-1.   0. ]
         [-0.5 -0.5]
         [-0.5  0. ]
         [-0.5  0.5]
         [ 0.  -1. ]
         [ 0.  -0.5]
         [ 0.   0. ]
         [ 0.   0.5]
         [ 0.   1. ]
         [ 0.5 -0.5]
         [ 0.5  0. ]
         [ 0.5  0.5]
         [ 1.   0. ]]
    """
    dimension = len(limits)
    if dimension == 1:
        return crange(limits[0][0], limits[0][1], sample_rates[0])
    elif dimension == 2:
        startx, stopx = limits[0]
        starty, stopy = limits[1]
        dx = sample_rates[0]
        dy = sample_rates[1]
        points = []
        for x in crange(startx, stopx, dx):
            for y in crange(starty, stopy, dy):
                point = np.array((x, y))
                if domain(point):
                    points.append(point)
        return np.array(points)
    elif dimension == 3:
        startx, stopx = limits[0]
        starty, stopy = limits[1]
        startz, stopz = limits[2]
        dx = sample_rates[0]
        dy = sample_rates[1]
        dz = sample_rates[2]
        points = []
        for x in crange(startx, stopx, dx):
            for y in crange(starty, stopy, dy):
                for z in crange(starty, stopy, dz):
                    point = np.array((x, y, z))
                    if domain(point):
                        points.append(point)
        return np.array(points)


def polargrid(limits, sample_rates, domain):
    """Returns points for a polar grid.

    Args:
        limits (list): limits for r and theta.
        sample_rates (list): sample rates for r and theta.
        domain (function): function for determining if point is in
                           the domain or not.
    Returns:
        numpy.ndarray: One dimensional list of all points in the domain.
    """
    points_in_polar = grid(limits, sample_rates, domain)
    points = []
    for r, teta in points_in_polar:
        points.append((r*math.cos(teta), r*math.sin(teta)))
    return points


def animate_1d_function(function, points, times):
    """Animates 1D function

    Args:
        function (function): function to animate
        points (numpy.ndarray): points to use in animation
        times (numpy.ndarray): time values to use in animation
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    def get_values(t):
        xs = []
        ys = []
        for point in points:
            xs.append(point)
            ys.append(function(point, t))
        return (xs, ys)
    xs, ys = get_values(0)
    ax.scatter(xs, ys, animated=True)

    def animate(t=0):
        xs, ys = get_values(t)
        scat = ax.scatter(xs, ys, animated=True)
        ax.set_title('t = ' + str(t) + 's')
        plt.draw()
        return scat,

    ani = animation.FuncAnimation(fig, animate,
                                  times, interval=50,
                                  repeat_delay=1000, blit=True)
    plt.show()


def animate_2d_function(function, points, times):
    """Animates 2D function

    Args:
        function (function): function to animate
        points (numpy.ndarray): points to use in animation
        times (numpy.ndarray): time values to use in animation
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = []
    ys = []
    zs = []
    # initialize values
    for point in points:
        xs.append(point[0])
        ys.append(point[1])
        zs.append(function(point, 0))
    a = ax.scatter(xs, ys, zs, animated=True)

    def animate(t=0):
        xs = []
        ys = []
        zs = []
        for point in points:
            xs.append(point[0])
            ys.append(point[1])
            zs.append(function(point, t))
        # a.set_array(np.array(zs))
        a._offsets3d = juggle_axes(xs, ys, zs, 'z')
        ax.set_title('t = ' + str(t) + 's')
        plt.draw()
        return a,

    ani = animation.FuncAnimation(fig, animate,
                                  times, interval=50,
                                  repeat_delay=1000, blit=True)
    plt.show()
