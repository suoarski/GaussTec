
import numpy as np

# Get euclidean distance of each pair of points
def getDistance(x, xk):
    difference = x.reshape(-1, 1) - xk.reshape(1, -1)
    return np.sqrt(difference**2)

# Gaussian radial basis function
def gaussRBF(radius, sigma, eps=1e8):
    if sigma.dtype != np.float64:
        sigma[sigma==0] = eps
    coeff = 1/(sigma*(2*np.pi)**0.5)
    return coeff * np.exp(-0.5*(radius / sigma)**2)