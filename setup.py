#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name="momentcalc",
    version="0.1.0",
    author="Ridvan Orsvuran",
    author_email="ridvan.orsvuran@gmail.com",
    description="Moment Calculator",
    license="GPLv3+",
    keywords="example documentation tutorial",
    url="http://rdno.org/momentcalc",
    packages=['moments', 'greens'],
    scripts=['scripts/calc_seismograms', 'scripts/parse_kineticout'],
    long_description="Moment Calculators for fault planes and utilities to work with green's functions.",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
)
