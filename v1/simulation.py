
# Code running simulations and saving them on disk

import numpy as np
import joblib
import sklearn.datasets as ds
import scipy.stats as stats
from SF import SFilter as SF

import data as D
import IT


simulation_ID = '20190823_'
n_simulations = 10
IT_bins=30

for simul_ctr in range(n_simulations):
    
    simulation_name = simulation_ID + str(simul_ctr)
    
    data = D.get_samples_multivariate_gaussians_ND(n_samples_train=900, n_samples_test=100, mu = stats.uniform.rvs(loc=-5,scale=10,size=4), sigma = ds.make_spd_matrix(4))
    joblib.dump(data, 'data_'+simulation_name)
    
    sf = SF()
    sf.setFeatures(8)
    sf.setTrainData(data['X_tr'])
    sf.initializeWeights()
    sf.initialiazeNotebook()
    sf.train()
       
    Ws = sf.notebook['weights']
    joblib.dump(Ws, 'weights_'+simulation_name)
    
    T_tes = []
    
    MI_XTs = []
    H_Ts = []
    MI_XTs_debug = []
    H_Ts_debug = []
    
    for i in range(len(Ws)):
        sf.setWeights(Ws[i])
        T_te = sf.feedForward(data['X_te'])
        T_tes.append(T_te)
        
        MI_XTs.append(IT.compute_MI_bin(data['X_te'],T_te,bins=IT_bins))
        H_Ts.append(IT.compute_H_bin(T_te,bins=IT_bins))
        
    results = {}
    results['T_te'] = T_tes
    results['MI_XT'] = MI_XTs
    results['H_T'] = H_Ts
    joblib.dump(results, 'results_'+simulation_name)


print('Done')

