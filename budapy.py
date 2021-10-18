#!/usr/bin/env python3
"""
Buddabrot from Mandelbrot fractal
"""
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from budcompute import mandelbrot, buddahbrot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 3000
a_min, a_max = -3, 2
b_max = 2
b_min = - b_max
L = a_max - a_min
da = L / N

a = np.arange(a_min, a_max, da)
b = np.arange(b_min, b_max, da)

M_b = np.zeros((b.shape[0], a.shape[0]), dtype=np.short)

A, B = np.meshgrid(a, b)
M_b = np.zeros_like(A)
if rank == 0:
    sumM_b = np.zeros((b.shape[0], a.shape[0]), dtype=np.short)
else:
    sumM_b = None

buddahbrot(a, b, maxit=20, N=1e8, M=M_b)

comm.Ireduce(M_b, sumM_b, op=MPI.SUM, root=0)

if rank == 0:
    M_b += M_b[::-1, :]
    # M_m = mandelbrot(a, b, 200)
    # M_m[M_m == 99] = 0
    # M = M_m + M_b
    np.save('budd.npy', M_b)
    M_b = np.load('budd.npy')
    plt.pcolormesh(M_b, cmap="inferno", shading="gouraud")
    plt.axis("image")
    plt.axis("off")
    plt.savefig("buddbrot.png", bbox_inches='tight', dpi=300)
    plt.show()
