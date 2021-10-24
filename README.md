# budapy
Generate beautiful figures based on the Mandelbrot fractal via parallel cumputing. 

This is not optimized, I have implemented a way faster and optimized version in c : [budack](https://github.com/Tugdual-G/budack).
    
However, this is a work in progress, python can be fast too.

![alt text](buddbrot.png)

Use mpi4py and numba.
Lauch with:

    mpiexec -n 4 python budapy.py
