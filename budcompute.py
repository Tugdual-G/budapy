#!/usr/bin/env python3
"""
buddabrot from Mandelbrot fractal
"""

import numpy as np
from numpy.random import random
from numba import jit, prange


@jit(nopython=True, parallel=True)
def mandelbrot(a, b, maxit, M):
    (l,) = a.shape
    (m,) = b.shape
    for i in prange(m):
        for j in prange(l):
            x0 = a[j]
            y0 = b[i]
            x2 = 0
            y2 = 0
            n = 0
            x = 0
            y = 0
            while x2 + y2 < 4 and n < maxit:
                y = 2 * x * y + y0
                x = x2 - y2 + x0
                x2 = x ** 2
                y2 = y ** 2
                n += 1
            M[i, j] = n


@jit(nopython=True)
def buddahbrot(a, b, maxit, N, M):
    dx = a[1] - a[0]
    (m,) = a.shape
    (l,) = b.shape
    nbrpts = 0
    xy = np.empty(maxit*2, dtype=np.short)

    while nbrpts < N:
        x0 = (a[-1] - a[0]) * random() + a[0]
        y0 = b[-1] * random()
        n = 0
        x = x0
        y = y0
        x2 = x0 ** 2
        y2 = y0 ** 2
        diverge = False
        noloop = True
        n = 0
        while n < maxit and not diverge and noloop:
            ix = int((x - a[0]) // dx)
            iy = int((y - b[0]) // dx)
            xy[n*2] = ix
            xy[n*2+1] = iy
            y = 2 * x * y + y0
            x = x2 - y2 + x0
            x2 = x ** 2
            y2 = y ** 2
            diverge = not (a[0] < x < a[-1] and b[0] < y < b[-1])
            n += 1
        if diverge:
            nbrpts += 1
            for i in range(0, n, 2):
                M[xy[i+1], xy[i]] += 1
