import numpy as np
import matplotlib.pyplot as plt
from temporal_schemes_v2 import (step_euler, step_inverse_euler, step_crank_nicholson, step_rk4, Leap_Frog)


def refine_time_grid(t):
   
    t_new = np.zeros(2*len(t) - 1)
    t_new[0::2] = t
    t_new[1::2] = 0.5*(t[:-1] + t[1:])
    return t_new


def mesh_error(F, scheme, t):
   
    U1 = cauchy_solve(F, scheme, t)             # malla gruesa
    t2 = refine_time_grid(t)
    U2 = cauchy_solve(F, scheme, t2)            # malla fina

    # comparación: tomar los puntos coincidentes
    U2_coarse = U2[::2]

    err = np.linalg.norm(U1 - U2_coarse, axis=1)
    return err[-1]     # error final global


def cauchy_solve(F, U0, scheme, t):
    dt = t[1] - t[0]
    U = np.zeros((len(t), len(U0)))
    U[0] = U0
    for n in range(len(t)-1):
        U[n+1] = scheme(F, U[n], t[n], dt)
    return U

def convergence_rate(F, scheme, U0, t0, tf, levels=6):
  

    logN = []
    logE = []

    # Malla inicial
    t = np.linspace(t0, tf, 20)

    for _ in range(levels):
        N = len(t) - 1
        error = mesh_error(F, scheme, t)

        logN.append(np.log10(N))
        logE.append(np.log10(error))

        # Refina malla
        t = refine_time_grid(t)

    # Ajuste lineal
    logN = np.array(logN)
    logE = np.array(logE)
    slope, intercept = np.polyfit(logN, logE, 1)

    print(f"Orden estimado: {abs(slope):.4f}")

    return logN, logE, slope


def compare_schemes(F, U0, t0, tf):
    methods = {
        "Euler"           : step_euler,
        "Inverse Euler"   : step_inverse_euler,
        "Crank–Nicolson"  : step_crank_nicholson,
        "RK4"             : step_rk4
    }

    plt.figure(figsize=(7,5))

    for name, method in methods.items():
        logN, logE, slope = convergence_rate(F, method, U0, t0, tf)
        plt.plot(logN, logE, marker="o", label=f"{name} (p≈{abs(slope):.2f})")

    plt.xlabel("log₁₀(N)")
    plt.ylabel("log₁₀(Error)")
    plt.title("Temporal Convergence Rate")
    plt.grid(True)
    plt.legend()
    plt.show()
