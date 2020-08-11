
# Code generating the data

import numpy as np
import scipy.stats as stats
import joblib
from matplotlib import pyplot as plt
from datageneration import data_manipulation


def get_samples_independent_gaussians_2D(n_samples_train=100, n_samples_test=10, 
                                         mu1 = 0.0, sigma1 = .5, mu2 = 0.0, sigma2 =.5,
                                         show=False, save=False, savename='data.pkl'):
     
    # Define the distributions along each dimension
    pdf_X_train_1 = stats.norm(loc = mu1, scale = sigma1)
    pdf_X_train_2 = stats.norm(loc = mu2, scale = sigma2)
    
    pdf_X_test_1 = stats.norm(loc = mu1, scale = sigma1)
    pdf_X_test_2 = stats.norm(loc = mu2, scale = sigma2)
    
    # Generate training samples
    samples_X_train_1 = pdf_X_train_1.rvs(n_samples_train)
    samples_X_train_2 = pdf_X_train_2.rvs(n_samples_train)
    samples_X_train = np.array([samples_X_train_1, samples_X_train_2]).T
    
    # Generate test samples
    samples_X_test_1 = pdf_X_test_1.rvs(n_samples_test)
    samples_X_test_2 = pdf_X_test_2.rvs(n_samples_test)
    samples_X_test = np.array([samples_X_test_1, samples_X_test_2]).T

    # Assemble data
    data = {}
    data['X_tr'] = samples_X_train[:,0:2]
    data['X_te']= samples_X_test[:,0:2]
    
    if(show):
        data_manipulation.display_XY_in_2D(data)
        plt.show()
        
    if(save):
        joblib.dump(data, savename)
        
    return data


def get_samples_multivariate_gaussians_ND(n_samples_train=100, n_samples_test=10, 
                                          mu = np.array([0,0]), sigma = np.array([[1,0],[0,1]]), 
                                         show=False, save=False, savename='data.pkl'):
    
    samples_X_train = stats.multivariate_normal.rvs(mean=mu, cov=sigma, size=n_samples_train)
    samples_X_test = stats.multivariate_normal.rvs(mean=mu, cov=sigma, size=n_samples_test)

    # Assemble data
    data = {}
    data['X_tr'] = samples_X_train
    data['X_te']= samples_X_test
    
    if(show):
        data_manipulation.display_XY_in_2D(data)
        plt.show()
        
    if(save):
        joblib.dump(data, savename)
        
    return data

    
    


