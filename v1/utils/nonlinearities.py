
import numpy as np

def sigmoid(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix after element-wise application of sigmoid
    """
    Y = 1.0 / (1.0 + np.exp(-X))
    return Y

def deriv_sigmoid(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix giving the element-wise derivation of sigmoid(X) wrt X
    """
    Y = (1.0 - sigmoid(X)) * sigmoid(X)
    return Y

def positivecosine(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix after element-wise application of cosine
    """
    Y = 1.0 + np.cos(X)
    return Y

def deriv_positivecosine(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix giving the element-wise derivation of cosine(X) wrt X
    """
    Y = -np.sin(X)
    return Y


def positivesine(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix after element-wise application of sine
    """
    Y = 1.0 + np.sin(X)
    return Y

def deriv_positivesine(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix giving the element-wise derivation of sin(X) wrt X
    """
    Y = np.cos(X)
    return Y

def negexp(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix after element-wise application of sine
    """
    Y = np.exp(-X)
    return Y

def deriv_negexp(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix giving the element-wise derivation of sin(X) wrt X
    """
    Y = -np.exp(-X)
    return Y



def softReLU(X,epsilon=0):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix after element-wise application of rectifier linear unit
    """
    Y = X.copy()
    Y[Y<0] = epsilon
    return Y

def deriv_softReLU(X,epsilon=0):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix giving the element-wise derivation of softReLU(X) wrt X
    """
    Y = np.ones(X.shape)
    Y[X<0] = 0
    return Y


def softabsolute(X, epsilon=1e-8):
    """
    INPUT:
    X: a numpy matrix
    epsilon: epsilon machine to compute soft-absolute value 
    
    OUTPUT:
    A numpy matrix after element-wise application of soft-absolute value
    """
    Y = np.sqrt(X**2 + epsilon) 
    return Y

def deriv_softabsolute(X, epsilon=1e-8):
    """
    INPUT:
    X: a numpy matrix
    epsilon: epsilon machine to compute soft-absolute value 
    
    OUTPUT:
    A numpy matrix giving the element-wise derivation of softabsolute(X) wrt X
    """
    Y = X / softabsolute(X, epsilon)
    return Y

def square(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix after element-wise application of power two
    """
    Y = X ** 2
    return Y

def deriv_square(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix giving the element-wise derivation of square(X) wrt X
    """
    Y = 2 * X
    return Y

def null(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix after element-wise application of null
    """
    Y = X
    return Y

def deriv_null(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix giving the element-wise derivation of null(X) wrt X
    """
    Y = 1
    return Y

def logarithm(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix after element-wise application of log
    """
    Y = np.log(X)
    return Y

def deriv_logarithm(X):
    """
    INPUT:
    X: a numpy matrix 
    
    OUTPUT:
    A numpy matrix giving the element-wise derivation of logarithm(X) wrt X
    """
    Y = 1/X
    return Y