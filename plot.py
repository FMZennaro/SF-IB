
# Code analyzing the simulations and generating the plots

import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.cm import get_cmap 

def getfile(filename):
    return os.path.join(os.getcwd(),'results',filename)

simulation_ID = '20190823BBB_'
n_simulations = 10

cmap = get_cmap('gist_rainbow')
colors = np.array([cmap(i) for i in np.linspace(0, 1, n_simulations)])
#colors = np.reshape(colors,[colors.shape[0],1,colors.shape[1]])
dotsize = 75

### LOAD DATA
nats2bits = 1.0/np.log(2)
list_MI_XT = []; list_H_T = []; list_deltaweight =[]
for i in range(n_simulations):    
    data = joblib.load(getfile('data_'+simulation_ID+str(i)))
    results = joblib.load(getfile('results_'+simulation_ID+str(i)))
    weights = joblib.load(getfile('weights_'+simulation_ID+str(i)))
    
    list_MI_XT.append(np.array(results['MI_XT'][1:]) * nats2bits)
    list_H_T.append(np.array(results['H_T'][1:]) * nats2bits)
    list_deltaweight.append(np.array([np.linalg.norm((weights[i]-weights[i+1])) for i in range(len(weights)-1)]))


### PLOT STANDARD GRAPHS
fig,axes = plt.subplots(nrows=1, ncols=3)
axes[0].set_title('I(X;T)'); axes[1].set_title('H(T)'); axes[2].set_title('Delta weights')
axes[0].set_xlabel('Iterations'); axes[1].set_xlabel('Iterations');  axes[2].set_xlabel('Iterations')
axes[0].set_ylabel('Bits'); axes[1].set_ylabel('Bits'); axes[2].set_ylabel('L2-Norm')

for i in range(n_simulations):
    axes[0].plot(list_MI_XT[i],c=colors[i])
    axes[1].plot(list_H_T[i],c=colors[i])
    axes[2].plot(list_deltaweight[i], c=colors[i])
plt.show()


### COMPUTE AVERAGE PER ITERATION
# Find the lowest common number of iterations
min_iter = np.min([list_MI_XT[i].shape[0] for i in range(len(list_MI_XT))])
# Reduce the data only to the common number of iterations
min_list_MI_XT = [list_MI_XT[i][0:min_iter] for i in range(len(list_MI_XT))]
min_list_H_T = [list_H_T[i][0:min_iter] for i in range(len(list_H_T))]
# Convert the data into a numpy array
min_mat_MI_XT = np.vstack(min_list_MI_XT)
min_mat_H_T = np.vstack(min_list_H_T)
# Compute averages over iterations
avg_list_MI_XT = np.mean(min_mat_MI_XT,axis=0)
avg_list_H_T = np.mean(min_mat_H_T,axis=0)

print(avg_list_H_T)
print(avg_list_MI_XT)


### PLOT INFORMATION GRAPH
fig,axes = plt.subplots(nrows=1, ncols=2)
axes[0].set_title('Information Graph (individual simulations)'); axes[0].set_xlabel('I(X;T)'); axes[0].set_ylabel('H(T)');
for i in range(n_simulations):
    axes[0].scatter(list_MI_XT[i],list_H_T[i],c=colors[i])
    axes[0].plot(list_MI_XT[i],list_H_T[i],c=colors[i])

colors = ['green'] + ['black']*(avg_list_MI_XT.shape[0]-2) + ['red']
dotsizes = [75] + [30]*(avg_list_MI_XT.shape[0]-2) + [75]
axes[1].set_title('Information Graph (average of all simulations)'); axes[1].set_xlabel('I(X;T)'); axes[1].set_ylabel('H(T)');    
axes[1].plot(avg_list_MI_XT,avg_list_H_T,c='black')
axes[1].scatter(avg_list_MI_XT,avg_list_H_T,c=colors,s=dotsizes)
plt.show()
