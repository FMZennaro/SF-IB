
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as kb
from tensorflow.keras.callbacks import LambdaCallback

def L1loss(y_true,y_pred):
    return kb.sum(y_pred)

class SFLayer(tf.keras.layers.Layer):
    def __init__(self,dim,epsilon=10**-8):
        self.dim=dim
        self.epsilon = tf.constant(epsilon)
        super(SFLayer,self).__init__()
        
    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(self.dim,input_shape[1]), initializer='normal', trainable=True)
        super(SFLayer,self).build(input_shape)
    
    def l2row(self,X):
        N = kb.sqrt(kb.sum(X**2, axis=1) + self.epsilon)
        Y = kb.transpose(kb.transpose(X) / N)
        return Y
    
    def softabsolute(self,X):
        return kb.sqrt(X**2 + self.epsilon)
           
    def call(self, x_input):
        X = kb.transpose(x_input)
        linWX = kb.dot(self.W,X)
        F = self.softabsolute(linWX)
        Fsquish = self.l2row(F)
        Fhat = self.l2row(kb.transpose(Fsquish))
        
        return Fhat

class SFilter():
    def __init__(self,n_features):
        self.n_features = n_features
        self.model = tf.keras.Sequential()
        self.model.add(SFLayer(n_features))
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=optimizer,loss=L1loss)
        
        self.weights = []
        self.Wcallback = LambdaCallback(on_epoch_end=lambda batch, logs: self.weights.append(self.model.layers[0].get_weights()[0]))
    
    def fit(self,X,epochs=200,verbose=1,batchsize=None):
        self.hist = self.model.fit(X,np.ones((X.shape[0],1)), epochs=epochs, callbacks = [self.Wcallback], verbose=verbose, batch_size = batchsize)
    
    def transform(self,X):
        return self.model.call(X)