
# Code implementing Sparse Filtering
# Code based on https://github.com/jngiam/sparseFiltering


import numpy as np
import joblib
from scipy.optimize import minimize


def l2row(X):    
    N = np.sqrt((X**2).sum(axis=1)+1e-8)
    Y = (X.T/N).T
    return Y,N
     
def l2rowg(X,Y,N,D):
    return (D.T/N - Y.T * (D*X).sum(axis=1) / N**2).T

def softabsolute(X, epsilon=1e-8):
    return np.sqrt(X**2 + epsilon) 

def deriv_softabsolute(X, epsilon=1e-8):
    return X / softabsolute(X, epsilon)

class SFilter(object):    
    
    ### INITIALIZERS ###
    def __init__(self, epochs=500, nonlinearity=softabsolute, deriv_nonlinearity=deriv_softabsolute):
        self.name = 'PSF'
        self.epochs = epochs
        self.nonlinearity = nonlinearity
        self.deriv_nonlinearity = deriv_nonlinearity
        self.notebook = {}
        
    def setFeatures(self,F):
        self.learned_features = F
                
    def setTrainData(self,X_tr):
        # It receives data in the external shape [samples x features] and it sets parameters in the
        # internal shape [features x samples]
        self.data_nsamples = X_tr.shape[0]
        self.original_features = X_tr.shape[1]
        self.data = X_tr.T
        
    def setWeights(self,W):
        self.W = W.copy()
                
    def initializeWeights(self):
        self.W = np.random.randn(self.learned_features,self.original_features)
    
    def initialiazeNotebook(self):
        self.notebook['weights'] = []
        self.notebook['losses'] = []
        
             
            
    ### TRAINING ###
    def train(self):
               
        optW = self.W
    
        def objFun(W):
            ## Feed forward
            W = W.reshape((self.learned_features,self.original_features))
            
            # Compute Z
            linWX = np.dot(W,self.data)
            F = self.nonlinearity(linWX)
            Fsquish, L2Fs = l2row(F)
            Fhat, L2Fn = l2row(Fsquish.T)
            
            # Record iteration
            self.callbackIteration(W)
                        
            ## Derivative of Sparse Filtering Loss Function
            ## Backprop through each feedforward step
            DeltaW = l2rowg(Fsquish.T, Fhat, L2Fn, np.ones(Fhat.shape))
            DeltaW = l2rowg(F, Fsquish, L2Fs, DeltaW.T)
            DeltaW = ((DeltaW*(self.deriv_nonlinearity(linWX))).dot(self.data.T))
            
            return Fhat.sum(), DeltaW.flatten()
                
        # Optimization
        self.current_iteration = 0
        _,_ = objFun(optW)
        res = minimize(objFun, optW, method='L-BFGS-B', jac = True, options = {'maxiter':self.epochs, 'disp':False})
        self.W = res.x.reshape(self.learned_features,self.original_features)

    def callbackIteration(self,W):
        self.notebook['weights'].append(W.copy())
        
        WX = np.dot(W, self.data)
        F = self.nonlinearity(WX)        
        Fsquish = l2row(F)[0]
        self.notebook['losses'].append(l2row(Fsquish.T)[0].sum())   
        return None
    
    def writeNotebook(self,filename):
        joblib.dump(self.notebook,filename) 
         
                             
    ### FEEDFORWARD MODULE ###           
    def feedForward(self,data):
        # This function is an external function
        # It receives data in the external shape [samples x features] and it returns results in the same
        # external shape [samples x features]
        WX = np.dot(self.W, data.T)
        F = self.nonlinearity(WX)
        
        Fsquish = l2row(F)[0]
        return l2row(Fsquish.T)[0]
    
       
    