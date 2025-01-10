import matplotlib.pyplot as plt
import numpy as np

def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N



N = 11
X = np.linspace(-2, 2, 11)
Y = np.linspace(-2, 2, 11)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 0.])
Sigma = np.array([[ 0.1 , 0], [0,  0.1]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

data = multivariate_gaussian(pos, mu, Sigma)



# Plot heatmap
plt.imshow(data, cmap='viridis', interpolation='nearest')
plt.colorbar()
# plt.savefig('reward_heatmap.pdf', bbox_inches='tight')
plt.show()
