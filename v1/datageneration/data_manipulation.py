
# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt




def cond(X,Y,y):
    return X[Y==y]

def stackdata(X):
    # Stack 2+D data along the columns
    # X1 (250 x 2), X2 (250 x 2)
    # X (500, 2)
    return np.vstack(X)

def stacklabel(Y):
    # Stack 1D labels along the columns
    # Y1 (250,), Y2 (250,)
    # Y (500,)
    return np.hstack(Y)

def assemble_plot_data(DATA):
    X = stackdata([DATA['X_tr'],DATA['X_te']])
    Y = stacklabel([DATA['Y_tr'],DATA['Y_te']])
    
    plot_data = {}
    plot_data['X'] = X
    plot_data['X_tr'] = DATA['X_tr']
    plot_data['X_te'] = DATA['X_te']
    plot_data['X_given_Y_1'] = cond(X,Y,1)
    plot_data['X_given_Y_0'] = cond(X,Y,0)
    plot_data['X_tr_given_Y_1'] = cond(DATA['X_tr'], DATA['Y_tr'], 1)
    plot_data['X_tr_given_Y_0'] = cond(DATA['X_tr'], DATA['Y_tr'], 0)
    plot_data['X_te_given_Y_1'] = cond(DATA['X_te'], DATA['Y_te'], 1)
    plot_data['X_te_given_Y_0'] = cond(DATA['X_te'], DATA['Y_te'], 0)
    
    return plot_data

def display_XY_in_2D(DATA,xlims=None, ylims=None):
    plot_data = assemble_plot_data(DATA)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, title="Samples in 2D")
    
    if(xlims==None):
        xlims = [DATA['X_1_min']-1, DATA['X_1_max']+1]
    if(ylims==None):
        ylims = [DATA['X_2_min']-1, DATA['X_2_max']+1]
    
    ax.scatter(plot_data['X_tr_given_Y_1'][:,0], plot_data['X_tr_given_Y_1'][:,1], c='blue', marker='x', label='Positive Train Samples (SF{X_train} | Y=1)')
    ax.scatter(plot_data['X_tr_given_Y_0'][:,0], plot_data['X_tr_given_Y_0'][:,1], c='blue', marker='o', label="Negative Train Samples (SF{X_train_1} | Y=0)")
    ax.scatter(plot_data['X_te_given_Y_1'][:,0], plot_data['X_te_given_Y_1'][:,1], c='red', marker='x', label="Positive Test Samples (SF{X_test_1} | Y=1)")
    ax.scatter(plot_data['X_te_given_Y_0'][:,0], plot_data['X_te_given_Y_0'][:,1], c='red', marker='o', label="Negative Test Samples (SF{X_test_1} | Y=0)")
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    ax.set_xlabel('First dimension')
    ax.set_ylabel('Second dimension')
           
    return fig    
         
    
    
    
    
    






