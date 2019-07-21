import numpy as np
   
def UKF(x0, P0, y, u, Q, R, h_func, f_func, N, tau):
    nmeas = y.shape[0]
    states = x0.shape[0]

    x_k = x0
    P_k = P0
    sigma_points = np.zeros((2 * N + 1, states, 1))
    alpha = np.zeros((2 * N + 1, 1))
    ysigma = np.zeros((2 * N + 1, y.shape[1]))

    x_hist = np.zeros((nmeas + 1, states, 1))
    P_hist = np.zeros((nmeas + 1, states, states))

    for k in range(0, nmeas):
        xpred = np.zeros(x_k.shape)
        Ppred = Q
        ypred = np.zeros(y[0,:].shape)
        P_y = np.zeros((ypred.shape[0], ypred.shape[0]))

        # Distribute sigma points
        L = np.linalg.cholesky(P_k)
        sigma_points = getSigmaPoints(L, N, tau, x_k, states)

        # Propagate sigma points
        for i in range(0, 2*N + 1):
            sigma_points[i] = f_func(sigma_points[i], P_k, u[k], jacobian=False)

        # Compute predicted mean and covariance
        alpha[0] = tau / (N + tau)
        alpha[1::] = 0.5 / (N + tau)

        xpred = estimateFromSigmapoints(sigma_points, alpha, Q, xpred, N, covariance=False)
        Ppred = estimateFromSigmapoints(sigma_points, alpha, Q, xpred, N, covariance=True)    

        x_hist[k, :, :] = xpred
        P_hist[k, :, :] = Ppred

        # Update sigma point
        L = np.linalg.cholesky(Ppred)
        sigma_points = getSigmaPoints(L, N, tau, xpred, states)

        # Predict measurements
        for i in range(0, 2 * N + 1):
            ysigma[i] = h_func(sigma_points[i], 0, jacobian=False)

        # Estimate mean and covariance of predicted measurements
        ypred = estimateFromSigmapoints(ysigma, alpha, R, ypred, N, covariance=False)
        P_y = estimateFromSigmapoints(ysigma, alpha, R, ypred, N, covariance=True) 

        # Compute cross covariance and Kalman gain
        P_xy = estimateCrossCovariance(sigma_points, ysigma, alpha, xpred, ypred, N)

        if P_y.shape[0] is 1:
            K_k = P_xy / P_y
        else:
            K_k = P_xy * np.linalg.inv(P_y)

        # Compute corrected mean and covariance
        x_k = xpred + K_k * (y[k] - ypred)
        P_k = Ppred + P_k * K_k * P_k.T

        x_hist[k, :, :] = x_k
        P_hist[k, :, :] = P_k
      
    return x_k, P_k


def getSigmaPoints(L, N, tau, x_k, states):
    sigma_points = np.zeros((2 * N + 1, states, 1))
    for i in range(0, N + 1):
        if i is 0:
            sigma_points[i] = x_k
        else:
            L_col = L[:, i - 1].reshape(states, 1)
            sigma_points[i] = x_k + np.sqrt(N + tau) * L_col
            sigma_points[i + N] = x_k - np.sqrt(N + tau) * L_col
    
    return sigma_points

def estimateFromSigmapoints(sigma_points, alpha, noise, predicted_value, N, covariance=False):
    if not covariance:
        retval = np.zeros(predicted_value.shape)
        for i in range(0, 2 * N + 1):
            retval += alpha[i] * sigma_points[i]
        
        return retval
    else:
        retval = noise
        for i in range(0, 2 * N + 1):
            retval += alpha[i] * (sigma_points[i] - predicted_value) * (sigma_points[i] - predicted_value).T
        
        return retval

def estimateCrossCovariance(sigma_points, ysigma, alpha, predicted_x, predicted_y, N):
    retval = np.zeros((predicted_x.shape[0], predicted_y.shape[0]))

    for i in range(0, 2 * N + 1):
        retval += alpha[i] * (sigma_points[i] - predicted_x) * (ysigma[i] - predicted_y).T
    
    return retval