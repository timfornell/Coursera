import numpy as np

def EKF(x0, P0, y, u, Q, R, h_func, f_func):
    nmeas = y.shape[0]
    states = x0.shape[0]

    x_k = x0
    P_k = P0
    x_hist = np.zeros((nmeas + 1, states, 1))
    P_hist = np.zeros((nmeas + 1, states, states))

    for k in range(0, nmeas):
        # Predict
        xpred = f_func(x_k, P_k, u[k], jacobian=False)
        F_k = f_func(x_k, P_k, 0, jacobian=True)
        Ppred = F_k @ P_k @ F_k.T + Q

        # Correction
        H_k = h_func(xpred, Ppred, jacobian=True)
        K_k = Ppred @ H_k.T @ np.linalg.inv(H_k @ Ppred @ H_k.T + R)
        x_k = xpred + K_k * (y[k] - h_func(xpred, Ppred, jacobian=False))
        P_k = (np.identity(states) - K_k @ H_k) @ Ppred

        x_hist[k,:,:] = x_k
        P_hist[k,:,:] = P_k

    return x_k, P_k