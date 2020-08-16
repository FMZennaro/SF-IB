
import numpy as np

def l2row(X):    
    N = np.sqrt((X**2).sum(axis=1)+1e-8)
    Y = (X.T/N).T
    return Y,N
    
    
def l2rowg(X,Y,N,D):
    """
    Backpropagate through Normalization.

    Parameters
    ----------

    X = Raw (possibly centered) data.
    Y = Row normalized data.
    N = Norms of rows.
    D = Deltas of previous layer. Used to compute gradient.

    Returns
    -------

    L2 normalized gradient.
    """
    return (D.T/N - Y.T * (D*X).sum(axis=1) / N**2).T

def scale_col(X):
    featmin = np.amin(X, axis=0)
    featmax = np.amax(X, axis=0)
    Xmin = np.tile(featmin,(X.shape[0],1))
    Xmax = np.tile(featmax,(X.shape[0],1))
    Xdelta = Xmax - Xmin
    Xdelta[Xdelta==0] = 1
    scaledX = np.divide(X - Xmin, Xdelta)

    return scaledX