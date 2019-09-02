
# Code implementing Sparse Filtering
# Code based on https://github.com/jngiam/sparseFiltering


import numpy as np
import joblib
from scipy.optimize import minimize
from utils import normalization, nonlinearities

class SFilter(object):    
    
    ### INITIALIZERS ###
    def __init__(self, iterations=500, nonlinearity=nonlinearities.softabsolute, deriv_nonlinearity=nonlinearities.deriv_softabsolute):
        self.name = 'PSF'
        self.iterations = iterations
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
                
        
    ### INITIALIZING WEIGHTS ###   
    def initializeWeights(self):
        self.W = np.random.randn(self.learned_features,self.original_features)
    
    def initialiazeNotebook(self):
        self.notebook['weights'] = []
        
             
            
    ### TRAINING ###
    def train(self):
               
        optW = self.W
    
        def objFun(W):
            ## Feed forward
            W = W.reshape((self.learned_features,self.original_features))
            
            # Compute Z
            linWX = np.dot(W,self.data)
            F = self.nonlinearity(linWX)
            Fsquish, L2Fs = normalization.l2row(F)
            Fhat, L2Fn = normalization.l2row(Fsquish.T)
            
            # Record iteration
            self.callbackIteration(W)
                        
            ## Derivative of Sparse Filtering Loss Function
            ## Backprop through each feedforward step
            DeltaW = normalization.l2rowg(Fsquish.T, Fhat, L2Fn, np.ones(Fhat.shape))
            DeltaW = normalization.l2rowg(F, Fsquish, L2Fs, DeltaW.T)
            DeltaW = ((DeltaW*(self.deriv_nonlinearity(linWX))).dot(self.data.T))
            
            return Fhat.sum(), DeltaW.flatten()
                
        # Optimization
        self.current_iteration = 0
        _,_ = objFun(optW)
        res = minimize(objFun, optW, method='L-BFGS-B', jac = True, options = {'maxiter':self.iterations, 'disp':False})
        self.W = res.x.reshape(self.learned_features,self.original_features)



    ### CALLBACK MODULE ###    
    def callbackIteration(self,W):
        self.notebook['weights'].append(W.copy())   
        return None 
         
                             
    ### FEEDFORWARD MODULE ###           
    def feedForward(self,data):
        # This function is an external function
        # It receives data in the external shape [samples x features] and it returns results in the same
        # external shape [samples x features]
        WX = np.dot(self.W, data.T)
        F = self.nonlinearity(WX)
        
        Fsquish = normalization.l2row(F)[0]
        return normalization.l2row(Fsquish.T)[0]
    
       
    def writeNotebook(self,filename):
        joblib.dump(self.notebook,filename)