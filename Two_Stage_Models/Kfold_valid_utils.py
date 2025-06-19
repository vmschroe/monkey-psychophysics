# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:03:48 2025

@author: schro
"""

from sklearn.metrics import mean_squared_error

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
    
    for j in range(K):
        testC = C_split[:,:,j]
        testN = N_split[:,:,j]
        
        trainC = np.dstack( (C_split[:,:,:j], C_split[:,:,j+1:]) )
        trainN = np.dstack( (N_split[:,:,:j], N_split[:,:,j+1:]) )
        trainC = np.sum(trainC, axis = 2)
        trainN = np.sum(trainN, axis = 2)
    
        if agg_sess == True:
            testC = np.sum(testC, axis = 0)
            testN = np.sum(testN, axis = 0)
            trainC = np.sum(trainC, axis = 0)
            trainN = np.sum(trainN, axis = 0)
        training_sets[j] = {'C': trainC, 'N': trainN}
        testing_sets[j] = {'C': testC, 'N': testN}
    
    return training_sets, testing_sets
        

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

def predict_c(fit_params, testN):
    p_pred = ffb.phi_with_lapses(fit_params,x)
    C_pred = p_pred*testN
    return C_pred

def predict_all_data(C,N,K=3):
    C_split, N_split = strat_K_split(C,N,K)
    training_sets, testing_sets = K_fold_train_test_split(C_split, N_split)
    Cpred = np.full((1,8),0)
    Ctrue = np.sum(C_split, axis=(0,2))
    N_after = np.sum(N_split, axis=(0,2))
    for j in range(K):
        train = training_sets[j]
        test = testing_sets[j]
        fit_params = fit_baysian_model(train['C'], train['N'])
        Cpred = Cpred + np.round(predict_c(fit_params, test['N']))
    return Cpred, Ctrue

def repeat_preds(data_dict, K=4, reps=10):
    C_preds = []
    C_trues = []
    for rep in range(reps):
        C_pred = {}
        C_true = {}
        for grp in ['ld','ln','rd','rn']:
            C_pred[grp], C_true[grp] = predict_all_data(C = data_dict[grp]['C_mat'], N = data_dict[grp]['N_mat'],K=3)
        
        C_pred = np.array(list(C_pred.values())).reshape((1,-1))
        C_true = np.array(list(C_true.values())).reshape((1,-1))
        C_preds.append(C_pred)
        C_trues.append(C_true)
    return np.array(C_preds), np.array(C_trues)

preds, trues = repeat_preds(data_dict)

# rmse = mean_squared_error(C_true.T,C_pred.T,squared=False)

# plt.scatter(C_true.reshape((1,-1)),C_pred.reshape((1,-1))-C_true.reshape((1,-1)))
# plt.hist(C_pred.reshape(32)-C_true.reshape(32))
