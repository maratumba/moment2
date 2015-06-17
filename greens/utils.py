# -*- coding: utf-8 -*-


def g_to_h(st):
    """Converts G to H
     Hnk,l = \int G_nk,l dt
    """
    tr = st[0]  # get first trace
    tr.integrate()
    return st


def derivative(st, n):
    """
    Takes derivate due to time n times
    """
    tr = st[0]
    for i in range(n):
        tr.differentiate()
    return st
