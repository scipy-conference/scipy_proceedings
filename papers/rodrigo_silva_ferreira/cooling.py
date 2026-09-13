# A small module written from the notebook -- source and notebook now coexist.
import numpy as np

def newton_cooling(T0, T_env, k, t):
    """Analytic solution of Newton's law of cooling: T(t)=T_env+(T0-T_env)e^{-kt}."""
    return T_env + (T0 - T_env) * np.exp(-k * np.asarray(t, dtype=float))

def euler_step(T, T_env, k, dt):
    """One explicit-Euler step of dT/dt = -k (T - T_env)."""
    return T + dt * (-k * (T - T_env))
