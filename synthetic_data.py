import numpy as np

def generate_synthetic_data(p, n, d, g):
    y = np.random.binomial(1, p, size=n)
    S = np.zeros((d, d))

    for i in range(d):
        for j in range(d):
            S[i, j] = g ** abs(i - j)
    
    mean_Y0 = np.zeros(d)
    mean_Y1 = np.array([1 / (i + 1) for i in range(d)])

    X = np.zeros((n, d))
    for i in range(n):
        if y[i] == 0:
            X[i] = np.random.multivariate_normal(mean_Y0, S)
        else:
            X[i] = np.random.multivariate_normal(mean_Y1, S)
    
    return X, y