
import itertools
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.callbacks import LambdaCallback

from tfSF import SFLayer

def Hloss(grid,sigma2):
    def loss(y_true,y_pred): 
        
        c1 = 1 / kb.sqrt(2*np.pi*sigma2)
        c2 = 2*sigma2

        kde = tf.map_fn( fn= lambda t: tf.reduce_sum(  c1 * kb.exp(-(y_pred[:,:-1]-t)**2 / c2 )), elems=grid) 
        
        px = kde / tf.reduce_sum(kde)
        
        return -kb.sum(px * kb.log(px))
    return loss


def Hloss_attempt(bins,linearizer,sigma2,vals):
    def loss(y_true,y_pred): 
        #self.linearizer = tf.constant(np.expand_dims(np.array([bins**i for i in range(n_features)]),axis=1),dtype='float')
        #self.vals = tf.constant(np.arange(0,bins**n_features),dtype='float')
                
        binsize = 1. / bins
        digitized = y_pred / binsize
        #digitized = tf.math.floordiv(y_pred,binsize)   
        linearized = kb.dot(digitized,linearizer)
        
        c1 = 1 / kb.sqrt(2*np.pi*sigma2)
        c2 = 2*sigma2

        px = tf.map_fn( fn= lambda t: tf.reduce_sum(  c1 * kb.exp(-(linearized-t)**2 / c2 )), elems=vals) 
        
        return kb.sum(px)
    return loss

def Hloss_orig(bins,linearizer):
    def loss(y_true,y_pred):
        binsize = 1. / bins
        digitized = tf.math.floor(y_pred / binsize)
        linearized = kb.dot(digitized,linearizer)
    
        _,_,unique_counts = tf.unique_with_counts( kb.squeeze(linearized,axis=1) )
        px = unique_counts / kb.sum(unique_counts)
    
        return -kb.sum(px * kb.log(px))
    return loss
          

class SFilterH():
    def __init__(self,n_features,bins=30, bandwidth=1./32):
        self.n_features = n_features
        self.model = tf.keras.Sequential()
        self.model.add(SFLayer(n_features))
        
        self.sigma2 = tf.constant(bandwidth,dtype='float')
        axis = np.linspace(0,1,bins)
        self.grid = tf.constant( np.array(list(itertools.product(axis,repeat=n_features-1))),dtype='float' )
                
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer,loss=Hloss(grid=self.grid,sigma2=self.sigma2))
        
        self.weights = []
        self.Wcallback = LambdaCallback(on_epoch_end=lambda batch, logs: self.weights.append(self.model.layers[0].get_weights()[0]))
    
    def fit(self,X,epochs=200,verbose=1,batchsize=None):
        self.hist = self.model.fit(X,np.ones((X.shape[0],1)), epochs=epochs, callbacks = [self.Wcallback], verbose=verbose, batch_size = batchsize)
    
    def transform(self,X):
        return self.model.call(X)