
import numpy as np
from numpy.linalg import norm


def step_euler(F, U, t, dt):
    """Un paso del método de Euler explícito."""
    return U + dt * F(U, t)


def step_inverse_euler(F, U, t, dt, maxiter=20, tol=1e-10):
    """Un paso del método de Euler inverso (implícito)."""
    U_new = U + dt * F(U, t)  # predictor
    for _ in range(maxiter):
        F_new = F(U_new, t + dt)
        U_next = U + dt * F_new
        if norm(U_next - U_new) < tol:
            break
        U_new = U_next
    return U_new


def step_crank_nicholson(F, U, t, dt, maxiter=20, tol=1e-10):
    """Un paso del método de Crank–Nicolson (implícito de segundo orden)."""
    U_new = U + dt * F(U, t)  # predictor inicial (Euler explícito)
    for _ in range(maxiter):
        F_n = F(U, t)
        F_next = F(U_new, t + dt)
        U_next = U + 0.5 * dt * (F_n + F_next)
        if norm(U_next - U_new) < tol:
            break
        U_new = U_next
    return U_new


def step_rk4(F, U, t, dt):
    """Un paso del método de Runge–Kutta clásico de cuarto orden."""
    k1 = F(U, t)
    k2 = F(U + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = F(U + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = F(U + dt * k3, t + dt)
    return U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def Leap_Frog (F, U, t, dt):

    x, v = U
    a = F(U, t)[1]    # aceleración actual


   
    v_half = v + 0.5 * dt * a


    x_next = x + dt * v_half

    a_next = F([x_next, v], t + dt)[1]


    v_next = v_half + 0.5 * dt * a_next

    return np.array([x_next, v_next])




__all__ = [
    "step_euler",
    "step_inverse_euler",
    "step_crank_nicholson",
    "step_rk4", "Leap_Frog", 
]



