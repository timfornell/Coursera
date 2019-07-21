from math import atan
import numpy as np
from EKF import EKF
from UKF import UKF

def f_func(x, P, u, jacobian=False):
    F = np.identity(2)
    F[0,1] = 0.5
    B = np.zeros((2,1))
    B[1,0] = 0.5

    if not jacobian:
        return F @ x + B * u
    else: 
        return F

def h_func(x, P, jacobian=False):
    S = 20
    D = 40
    p = x[0, 0]
    H = np.zeros((1, 2))
    H[0, 0] = S / (D - p)
    if not jacobian:
        return np.array(atan(H[0, 0]) * 180 / np.pi)
    else:
        H[0, 0] = S / ((D - p)*(D - p) + S * S)
        return H

y = np.zeros((1,1))
y[0] = 30
u = np.zeros((1,1))
u[0] = -2
R = 0.01
Q = np.identity(2) * 0.1

x0 = np.zeros((2, 1))
P0 = np.zeros((2, 2))
x0[1, 0] = 5
P0[0,0] = 0.05
P0[1,1] = 1

x, P = EKF(x0, P0, y, u, Q, R, h_func, f_func)

print("Got x: %s" % x.tolist())
print("Got P: %s" % P.tolist())

N = 2
tau = 3 - N

P0[0, 0] = 0.01

x, P = UKF(x0, P0, y, u, Q, R, h_func, f_func, N, tau)

print("Got x: %s" % x.tolist())
print("Got P: %s" % P.tolist())