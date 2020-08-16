
# Code for computing information-theoretic quantities
# Code based on https://github.com/ravidziv/IDNNs and https://github.com/artemyk/ibsgd

import numpy as np

def compute_H_bin(x,bins=30):
    binsize = 1. / bins
    digitized = np.floor(x / binsize).astype('int')
    px, _ = get_unique_probs(digitized)
    return -np.sum(px * np.log(px))
    

def compute_MI_bin(x,y,bins=30):
    p_xs, unique_inverse_x = get_unique_probs(x)
    
    bins = np.linspace(0, 1, bins, dtype='float32') 
    digitized = bins[np.digitize(np.squeeze(y.reshape(1, -1)), bins) - 1].reshape(len(y), -1)
    p_ts, _ = get_unique_probs( digitized )
    
    H_LAYER = -np.sum(p_ts * np.log(p_ts))
    H_LAYER_GIVEN_INPUT = 0.
    for xval in np.arange(len(p_xs)):
        p_t_given_x, _ = get_unique_probs(digitized[unique_inverse_x == xval, :])
        H_LAYER_GIVEN_INPUT += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x))
    return H_LAYER - H_LAYER_GIVEN_INPUT

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

