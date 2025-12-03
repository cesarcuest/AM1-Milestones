
import numpy as np

def newton_1d(g, x0, args=(), tol=1e-12, maxiter=50):
    x = x0
    for k in range(maxiter):
        fx = g(x, *args)
        if abs(fx) < tol:
            return x
        h = 1e-6
        dfx = (g(x+h, *args) - g(x-h, *args)) / (2*h)
        x = x - fx/dfx
    raise RuntimeError("Newton 1D no converge")