
# Newton 2d y JACOBIANO

import numpy as np



def newton_2d(f, X0, args=(), tol=1e-12, maxiter=50):
    X = np.array(X0, dtype=float)

    for k in range(maxiter):
        F = f(X, *args)
        if np.linalg.norm(F) < tol:
            return X

        # Jacobiano
        J = np.zeros((2,2))
        h = 1e-6
        for i in range(2):
            dX = np.zeros(2)
            dX[i] = h
            J[:, i] = (f(X + dX, *args) - f(X - dX, *args)) / (2*h)

        # Resolver J * delta = -F
        delta = np.linalg.solve(J, -F)
        X = X + delta

    raise RuntimeError("Newton no converge")
