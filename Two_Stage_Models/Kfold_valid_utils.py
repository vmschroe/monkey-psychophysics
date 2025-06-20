# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:03:48 2025

@author: schro
"""

from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D

plot_save = False


def strat_K_split(C,N,K=3, info = False):
    """
    splits binomial data into K stratified folds

    Parameters:
    C : np.array, shape=(Nsess, 8), counts of "high" response (each sess, stimAMP)
    N : np.array, shape=(Nsess, 8), number of trials (each sess, stimAMP)
    K : integer, number of groups
    
    Returns:
    C_split, N_split: np.arrays, shape=(Nsess, 8, K)

    """
    if C.shape != N.shape:
        print('Error: size mismatch between C and N matrices')
        return
    Nsess, _ = C.shape
    F = np.floor(N/K)
    C_split = np.full((Nsess,8,K),0) #np.array, shape=(Nsess, 8, K)
    N_split = np.dstack([F] * K) #np.array, shape=(Nsess, 8, K)
    for s in range(Nsess):
        for i in range(8):
            c = int(C[s,i])
            n = int(N[s,i])
            n_rounded = int(F[s,i])*K
            d = np.hstack([ np.full(c,1) , np.full(n-c,0) ])
            np.random.shuffle(d)
            D = d[0:n_rounded]
            C_split[s,i] = np.sum(D.reshape(K,int(F[s,i])), axis = 1)
    if info == True:
        lost = int(sum(sum(N))-sum(sum(sum(N_split))))
        print(f"Split {int(sum(sum(N)))} trials into {K} folds of {int(sum(sum(F)))} trials, stratified on stim. amp. and session")
        print(f"Cut {lost} trials, {round(100*lost/sum(sum(N)),1)}% of the data")
    return C_split, N_split
#  C_split, N_split = strat_K_split(C, N, K = 3, info = False)

def K_fold_train_test_split(C_split, N_split, agg_sess = True):
    """
    constructs K holdout/training sets from K stratified folds

    Parameters: 
    C_split : np.array, shape=(Nsess, 8, K), counts of "high" response (each sess, stimAMP, fold)
    N_split : np.array, shape=(Nsess, 8, K), number of trials (each sess, stimAMP, fold)
    agg_sess : bool, whether to aggregate session data or leave it stratified
    
    Returns:
    training_sets, testing_sets : dictionaries with keys 0:K, each containing dictionary with 'C' and 'N'
    """
    Nsess, _, K = C_split.shape
    if C_split.shape != N_split.shape:
        print('Error: size mismatch between C and N matrices')
        return
    
    training_sets = {}
    testing_sets = {}
    strat_train = {}
    strat_test = {}
    
    for j in range(K):
        testC = C_split[:,:,j]
        testN = N_split[:,:,j]
        
        trainC = np.dstack( (C_split[:,:,:j], C_split[:,:,j+1:]) )
        trainN = np.dstack( (N_split[:,:,:j], N_split[:,:,j+1:]) )
        trainC = np.sum(trainC, axis = 2)
        trainN = np.sum(trainN, axis = 2)
    
        if agg_sess == True:
            testCagg = np.sum(testC, axis = 0)
            testNagg = np.sum(testN, axis = 0)
            trainCagg = np.sum(trainC, axis = 0)
            trainNagg = np.sum(trainN, axis = 0)
        strat_train[j] = {'C': trainC, 'N': trainN}
        testing_sets[j] = {'C': testC, 'N': testN}
        training_sets[j] = {'C': trainC, 'N': trainN}
        testing_sets[j] = {'C': testC, 'N': testN}
    
    return training_sets, testing_sets, strat_data
# training_sets, testing_sets = K_fold_train_test_split(C_split, N_split, agg_sess = True)       

def fit_baysian_model(C, N, trace=False):
    with pm.Model() as model_pooled:
        # Define priors for the parameters
        W_gam = pm.Beta("W_gam",alpha=params_prior_params[0][0],beta=params_prior_params[0][1])
        gam = pm.Deterministic("gam", params_prior_scale[0]*W_gam)
        W_lam = pm.Beta("W_lam",alpha=params_prior_params[1][0],beta=params_prior_params[1][1])
        lam = pm.Deterministic("lam", params_prior_scale[1]*W_lam)
        W_b0 = pm.Gamma("W_b0",alpha=params_prior_params[2][0],beta=params_prior_params[2][1])
        b0 = pm.Deterministic("b0", params_prior_scale[2]*W_b0)
        W_b1 = pm.Gamma("b1_norm",alpha=params_prior_params[3][0],beta=params_prior_params[3][1])
        b1 = pm.Deterministic("b1", params_prior_scale[3]*W_b1)
        # Define PSE and JND as deterministic variables
        pse = pm.Deterministic("pse", ffb.PSE(gam, lam, b0, b1))
        jnd = pm.Deterministic("jnd", ffb.JND(gam, lam, b0, b1))
        # Define the likelihood
        likelihood = pm.Binomial("obs", n=N, p=ffb.phi_with_lapses([gam, lam, b0, b1],x), observed=C)
        
        #pm.Binomial("obs", n=N, p=theta, observed=data)
        # use Markov Chain Monte Carlo (MCMC) to draw samples from the posterior
        trace_pooled = pm.sample(1000, return_inferencedata=True, idata_kwargs={"log_likelihood": True})
    if trace:
        return trace_pooled
    fit_params = np.array(az.summary(trace_pooled)['mean'][['gam','lam','b0','b1']])
    return fit_params
# fit_params = fit_baysian_model(C, N, trace=False)

def predict_c(fit_params, testN):
    p_pred = ffb.phi_with_lapses(fit_params,x)
    C_pred = p_pred*testN
    return C_pred
# C_pred = predict_c(fit_params, testN)

def predict_all_data(C,N,K=3):
    C_split, N_split = strat_K_split(C,N,K)
    training_sets, testing_sets, strat_data = K_fold_train_test_split(C_split, N_split, agg_sess = True)
    Cpred = np.full((1,8),0)
    Ctrue = np.sum(C_split, axis=(0,2))
    N_after = np.sum(N_split, axis=(0,2))
    for j in range(K):
        train = training_sets[j]
        test = testing_sets[j]
        fit_params = fit_baysian_model(train['C'], train['N'])
        Cpred = Cpred + np.round(predict_c(fit_params, test['N']))
    return Cpred, Ctrue
# Cpred, Ctrue = predict_all_data(C,N,K=3)

def repeat_preds(data_dict, K=3, reps=10):
    C_preds = []
    C_trues = []
    for rep in range(reps):
        C_pred = {}
        C_true = {}
        for grp in ['ld','ln','rd','rn']:
            C_pred[grp], C_true[grp] = predict_all_data(C = data_dict[grp]['C_mat'], N = data_dict[grp]['N_mat'],K=K)
        
        C_pred = np.array(list(C_pred.values())).reshape((1,-1))
        C_true = np.array(list(C_true.values())).reshape((1,-1))
        C_preds.append(C_pred)
        C_trues.append(C_true)
    return np.array(C_preds), np.array(C_trues)
# preds_array, trues_array = repeat_preds(data_dict, K=3, reps=10)

preds, trues = repeat_preds(data_dict)

trues = trues.reshape((10,4,8))
preds = preds.reshape((10,4,8))


N_split = {}
for grp in ['ld','ln','rd','rn']:
    _, N_split[grp] = strat_K_split(C = data_dict[grp]['C_mat'], N = data_dict[grp]['N_mat'],K=3, info = True)

N_after = {}
for grp in ['ld','ln','rd','rn']:
    N_after[grp] = np.sum(N_split[grp],axis=(0,2))
    
Ns = np.array(list(N_after.values())).reshape((1,-1))

Ns = np.vstack([Ns]*10).reshape((10,4,8))

residuals = trues-preds
x = np.array(x)
res_ps=residuals/Ns
# Define colors for the four groups
colors = ['red', 'navy', 'pink', 'skyblue']
offset = np.array([-1.2,-.4,.4,1.2])
labels = ['left hand, distracted', 
          'left hand, not distracted', 
          'right hand, distracted', 
          'right hand, not distracted']


# Create scatter plot
if plot_save:
    plt.figure(figsize=(8, 6))
    for group in range(4):
        for i in range(8):
            temp_mean = np.mean(res_ps[:,group,i])
            temp_std = np.std(res_ps[:,group,i])
            plt.errorbar(x[i]+offset[group], temp_mean, yerr=temp_std, fmt = 'o', color = colors[group], elinewidth = 2)
    plt.hlines(0,4,52,colors='gray',linestyle=':')
    plt.xlabel('stim amp')
    plt.ylabel('residuals')
    plt.title('Error in Response Proportion Prediction: 3-Fold Cross-Validation of Bayesian Model')
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=labels[i],
               markerfacecolor=colors[i], markeredgecolor=colors[i], markersize=8,
               linestyle='None') for i in range(4)
    ]
    x_ticks = np.array([6, 12, 18, 24, 28, 32, 38, 44, 50])
    plt.xticks(x_ticks)
    plt.legend(handles=legend_handles, title="Mean ± SD")
    plt.tight_layout()
    plt.savefig(f'Bayes_3Fold_CV_plot.png')
    plt.show()
# rmse = mean_squared_error(C_true.T,C_pred.T,squared=False)

# plt.scatter(C_true.reshape((1,-1)),C_pred.reshape((1,-1))-C_true.reshape((1,-1)))
# plt.hist(C_pred.reshape(32)-C_true.reshape(32))
