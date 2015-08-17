# -*- coding: utf-8 -*-
import re


def g_to_h(st):
    """Converts G to H
     Hnk,l = \int G_nk,l dt
    """
    tr = st[0]  # get first trace
    tr.integrate()
    return st


def h_to_g(st):
    """Convert H to G
     Gnk,l = d G_nk,l / dt
    """
    tr = st[0]  # get first trace
    tr.differentiate()
    return st


def derivative(st, n):
    """
    Takes derivate due to time n times
    """
    tr = st[0]
    for i in range(n):
        tr.differentiate()
    return st


def get_greens_name(m, n, component, terms, for_force=False):
    """Return corresponding green's function for moment

    Args:
        moment_index (tuple): ((m,n), (indices))
        m (int): spatial moment
        n (int): temporal moment
        component (int): component (0=x, 1=y, 2=z, 3=n)
        terms (numpy.ndarray): other terms
        for_force (bool): use force formulation instead of moment tensor

    Returns:
        str: Green's function's name
    """
    c = "xyzn"
    # for force formulation use G, for moment tensor use H
    f_name = "G" if for_force else "H"
    comp = c[component]
    subs = "".join([c[term] for term in terms])
    name = "{G}_{comp}{subs}".format(G=f_name,
                                     comp=comp,
                                     subs=subs)

    # if name has more than 2 subscripts add a comma
    if len(name) > 4:
        name = name[:4] + "," + name[4:]

    # if there is a time derivative
    dt = "dt" if for_force else "dT"
    if n == 1:
        name = "{dt} ".format(dt=dt) + name
    if n > 1:
        name = "{dt}^{n} ".format(dt=dt, n=n) + name

    return name


def parse_greens_name(greens_name):
    parser = re.compile('(d[tT](\^(?P<n>\d+))? )?(?P<name>\w)_(?P<comp>\w)(?P<terms>[\w,]*)')
    match = parser.match(greens_name)
    n = match.group("n") or 0
    # special case for dt G..., n => 1
    if n == 0 and match.group(1) is not None:
        n = 1
    is_h = match.group("name") == "H"
    comp = match.group("comp")
    terms = match.group("terms").replace(",", "")
    zero_terms = 2 if is_h else 1
    m = len(terms) - zero_terms
    data = {
        "m": m,
        "n": int(n),
        "is_h": is_h,
        "comp": comp,
        "terms": terms,
    }
    return data


def get_greens_filename(greens_name, station_name):
    """Return green's function's filename
    """
    data = parse_greens_name(greens_name)
    return "{name}_{terms}.{comp}".format(name=station_name,
                                          terms=data['terms'],
                                          comp=data['comp'])


def _base_symmetry_for_derivatives(terms):
    i = terms[0]
    rest = terms[1:]
    # turn i,yxz; i,zxy and i,xyz to i,xyz
    return i+''.join(sorted(rest))


def are_greens_same(name1, name2):
    """Check if two green's functions are equal.

    It also checks for symmetry.

    Args:
        name1 (str): first green's function's name
        name2 (str): second green's function's name

    Returns:
        bool: equality of green's functions
    """

    params1 = parse_greens_name(name1)
    params1['terms'] = _base_symmetry_for_derivatives(params1['terms'])

    params2 = parse_greens_name(name2)
    params2['terms'] = _base_symmetry_for_derivatives(params2['terms'])

    return params1 == params2


def are_greens_equivalent(name1, name2):
    """Check for special case of green's functions

    It returns true if green's functions are the same.

    Special case:
       H_ni,jab == H_nj,iab

    This is done because moment tensor M is symmetric, so their
    coefficients for Mij and Mji is the same.

    Args:
        name1 (str): first green's function's name
        name2 (str): second green's function's name

    Returns:
        bool: equivalence of green's functions
    """

    if are_greens_same(name1, name2):
        return True

    params1 = parse_greens_name(name1)
    params2 = parse_greens_name(name2)

    # Special case is not true for force formulation
    if not params1['is_h'] or not params2['is_h']:
        return False

    # H_ni,jab => H_nj,iab
    params1['terms'] = params1['terms'][1] + params1['terms'][0] + params1['terms'][2:]

    new_terms1 = _base_symmetry_for_derivatives(params1['terms'])
    new_terms2 = _base_symmetry_for_derivatives(params2['terms'])
    return new_terms1 == new_terms2
