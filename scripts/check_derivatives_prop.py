from functools import partial

import numpy as np
from robust.robust_python import (
    PreintegratedWheelMeasurements,
    PreintegratedWheelParams,
)
from scipy.linalg import expm

np.set_printoptions(suppress=True, precision=4, linewidth=300)

# ------------------------- HELPERS ------------------------- #
e1, e2, e3 = np.eye(3)
E = np.array([[0, -1, 0], [1, 0, 0]])
M = lambda v: np.array([[v[0], v[1], 0], [0, v[0], v[1]], [0, 0, 0]])


def nder(f, x, eps=1e-6):
    fx = f(x)
    N = x.shape[0]
    M = fx.shape[0]
    d = np.zeros((M, N))
    for i in range(N):
        temp = x.copy()
        temp[i] += eps
        d[:, i] = (f(temp) - fx) / eps

    return d


def cross(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def H(theta):
    norm = np.linalg.norm(theta)
    if norm < 1e-2:
        return np.eye(3)
    A = (1 - np.cos(norm)) / norm**2
    B = (norm - np.sin(norm)) / norm**3
    tx = cross(theta)
    return np.eye(3) - A * tx + B * tx @ tx


def make_omega_v(p, w, v, A, R):
    gradMnorm = np.linalg.norm(A @ p + e3)
    v_body = np.array([v, 0, 0])
    w_body_start = E @ R.T @ A @ R @ v_body / gradMnorm
    w_body = np.array([w_body_start[0], w_body_start[1], w])

    return w_body, v_body


def f(state, noise, w, v, dt):
    theta = state[:3]
    p = state[3:6]
    m = state[6:9]
    eta_w = noise[0]
    eta_v = noise[1]
    eta_m = noise[2:5]
    eta_minit = noise[5:8]

    ms = m - eta_minit
    A = np.array([[ms[0], ms[1], 0], [ms[1], ms[2], 0], [0, 0, 0]])

    R = expm(cross(theta))
    Ht = H(theta)
    Htinv = np.linalg.inv(Ht)

    w_body, v_body = make_omega_v(p, w - eta_w, v - eta_v, A, R)

    theta_plus = theta + Htinv @ w_body * dt
    p_plus = p + R @ v_body * dt
    m_plus = m + eta_m
    return np.concatenate((theta_plus, p_plus, m_plus))


def makeA(state, noise, w, v, dt):
    theta = state[:3]
    p = state[3:6]
    m = state[6:9]

    ms = m
    A = np.array([[ms[0], ms[1], 0], [ms[1], ms[2], 0], [0, 0, 0]])

    R = expm(cross(theta))
    Ht = H(theta)
    Htinv = np.linalg.inv(Ht)
    gradMnorm = np.linalg.norm(A @ p + e3)

    w_body, v_body = make_omega_v(p, w, v, A, R)

    D = np.eye(9)
    # D_R_R
    D[:3, :3] += -cross(w_body) * dt / 2
    # D_R_m
    D[:3, 6:9] = Htinv[:, :2] @ E @ R.T @ M(R @ v_body) * dt / gradMnorm
    print(Htinv[:, :2] @ E @ R.T @ M(R @ v_body))
    print(dt, gradMnorm)
    # D_p_theta
    D[3:6, :3] = -R @ cross(v_body) @ Ht * dt

    return D


def makeB(state, noise, w, v, dt):
    theta = state[:3]
    p = state[3:6]
    m = state[6:9]
    eta_w = noise[0]
    eta_v = noise[1]
    eta_m = noise[2:5]
    eta_minit = noise[5:8]

    ms = m - eta_minit
    A = np.array([[ms[0], ms[1], 0], [ms[1], ms[2], 0], [0, 0, 0]])

    R = expm(cross(theta))
    Ht = H(theta)
    Htinv = np.linalg.inv(Ht)
    gradMnorm = np.linalg.norm(A @ p + e3)

    w_body, v_body = make_omega_v(p, w, v, A, R)

    D = np.zeros((9, 8))
    # D_theta_etaw
    D[:3, 0] = -Htinv @ e3 * dt
    # D_theta_etav
    D[:3, 1] = -Htinv[:, :2] @ E @ R.T @ A @ R @ e1 * dt / gradMnorm
    # D_theta_minit
    D[:3, 5:8] = -Htinv[:, :2] @ E @ R.T @ M(R @ v_body) * dt / gradMnorm

    # D_p_eta_v
    D[3:6, 1] = -R @ e1 * dt

    # D_m_etam
    D[6:9, 2:5] = np.eye(3)

    return D


theta = np.array([0.02, 0.04, 0])
p = np.array([0, 0, 0])
a = np.array([1.0, 0.2, 0.5])
v = 1
w = 1
dt = 0.1

state = np.concatenate((theta, p, a))
noise = np.zeros(8)

print("A - Numerical")
print(Ander := nder(partial(f, noise=noise, v=v, w=w, dt=dt), state))
print("A - Analytical")
print(Aan := makeA(state, noise, w, v, dt))
print(np.linalg.norm(Ander - Aan))
print()


def f_noise(n):
    return f(state, n, w, v, dt)


print("B - Numerical")
print(Bnder := nder(f_noise, noise))
print("B - Analytical")
print(Ban := makeB(state, noise, w, v, dt))
print(np.linalg.norm(Bnder - Ban))
print()
