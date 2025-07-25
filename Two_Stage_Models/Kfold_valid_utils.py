# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:03:48 2025

@author: schro
"""

from sklearn.metrics import mean_squared_error
from matplotlib.lines import Line2D
from Three_Stage_Models.High3sLogFuncs import HighLogAnalysis
# from Two_Stage_Models import bayes4param_ppc as b4p

#%%

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

def K_fold_train_test_split(C, N, K=3, info = False):
    """
    constructs K holdout/training sets from K stratified folds

    Parameters: 
    C_split : np.array, shape=(Nsess, 8, K), counts of "high" response (each sess, stimAMP, fold)
    N_split : np.array, shape=(Nsess, 8, K), number of trials (each sess, stimAMP, fold)
    agg_sess : bool, whether to aggregate session data or leave it stratified
    
    Returns:
    ready_data = {
        'sess_pool': {
            'train': pool_train ~dict with K keys, each w dict with 'C' and 'N'
            'test': pool_test, 
            'all': all_pool}, ~dict with K keys, each w trunced data
        'sess_strat': {
            'train': strat_train, 
            'test':strat_test, 
            'all': all_strat}}
    
    """
    C_split, N_split = strat_K_split(C, N, K=K, info = info)
    
    Nsess, _, K = C_split.shape
    if C_split.shape != N_split.shape:
        print('Error: size mismatch between C and N matrices')
        #return
    
    pool_train = {}
    pool_test = {}
    strat_train = {}
    strat_test = {}
    C_all_strat = np.sum(C_split, axis = 2)
    N_all_strat = np.sum(N_split, axis = 2)
    C_all_pool = np.sum(C_split, axis = (0,2))
    N_all_pool = np.sum(N_split, axis = (0,2))
    all_pool = {'C': C_all_pool, 'N': N_all_pool}
    all_strat = {'C': C_all_strat, 'N': N_all_strat}
    for j in range(K):
        testC = C_split[:,:,j]
        testN = N_split[:,:,j]
        
        trainC = np.dstack( (C_split[:,:,:j], C_split[:,:,j+1:]) )
        trainN = np.dstack( (N_split[:,:,:j], N_split[:,:,j+1:]) )
        trainC = np.sum(trainC, axis = 2)
        trainN = np.sum(trainN, axis = 2)
    
        testCpool = np.sum(testC, axis = 0)
        testNpool = np.sum(testN, axis = 0)
        trainCpool = np.sum(trainC, axis = 0)
        trainNpool = np.sum(trainN, axis = 0)
        
        pool_train[j] = {'C': trainCpool, 'N': trainNpool}
        pool_test[j] = {'C': testCpool, 'N': testNpool}
        strat_train[j] = {'C': trainC, 'N': trainN}
        strat_test[j] = {'C': testC, 'N': testN}
       
    ready_data = {'sess_pool': {'train': pool_train, 'test': pool_test, 'all': all_pool}, 'sess_strat': {'train': strat_train, 'test':strat_test, 'all': all_strat}}
    return ready_data
# ready_data = K_fold_train_test_split(C, N, K=3, info = False) 
    #   ready_data = {'sess_pool': {'train': pool_train, 'test': pool_test, 'all': all_pool}, 
    #               'sess_strat': {'train': strat_train, 'test':strat_test, 'all': all_strat}}
    #   ready_data['sess_pool']['train'][j]['C']

def fit_bayesian_model(C, N, trace=False):
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
# fit_params = fit_bayesian_model(C, N, trace=False)

def fit_hier_model(C,N,trace = False):
    temp_dict = {}
    temp_dict['C_mat'] = C
    temp_dict['N_mat'] = N
    trace_strat = HighLogAnalysis(temp_dict, mu_xi = np.array([1.,4.,1.,4.,6.,1.,1.,5.]), sigma_xi = 4*np.array([1.,4.,1.,4.,6.,1.,1.,5.]))
    if trace:
        return trace_strat
    fit_params = np.array(az.summary(trace_strat, var_names = ['gamma_h', 'gamma_l', 'beta0', 'beta1'])['mean']).reshape((4,-1)).T
    return fit_params

def hier_preds(training_sets, testing_sets, K=3, Nsess=43):
    def predict_c(fit_params, testN):
        p_pred = np.array([])
        for s in range(Nsess):
            p_pred = np.append(p_pred,ffb.phi_with_lapses(fit_params[s,:],x))
        p_pred = p_pred.reshape((-1,8))
        C_pred = p_pred*testN
        return C_pred
    Cpred = np.full((Nsess,8),0)
    for j in range(K):
        print(f'~~~~~ {j+1} of {K} ~~~~~~')
        train = training_sets[j]
        test = testing_sets[j]
        fit_params = fit_hier_model(train['C'], train['N'])
        Cpred = Cpred + predict_c(fit_params, test['N'])
    return Cpred
    
    
def bayes_preds(training_sets, testing_sets, K=3):
    def predict_c(fit_params, testN):
        p_pred = ffb.phi_with_lapses(fit_params,x)
        C_pred = p_pred*testN
        return C_pred
    Cpred = np.full((1,8),0)
    for j in range(K):
        print(f'~~~~~ {j+1} of {K} ~~~~~~')
        train = training_sets[j]
        test = testing_sets[j]
        fit_params = fit_bayesian_model(train['C'], train['N'])
        Cpred = Cpred + predict_c(fit_params, test['N'])
    return Cpred
# Cpred = bayes_preds(training_sets, testing_sets, K=3)


def plot_KFoldCV(preds, trues, Ns, K, model_name, plot_save=False, ylims = [-0.06, 0.06]):
    residuals = trues-preds
    res_ps=residuals/Ns
    # Define colors for the four groups
    colors = ['red', 'navy', 'pink', 'skyblue']
    offset = np.array([-1.2,-.4,.4,1.2])
    labels = ['left hand, distracted', 
              'left hand, not distracted', 
              'right hand, distracted', 
              'right hand, not distracted']
    
    
    plt.figure(figsize=(8, 6))
    for group in range(4):
        for i in range(8):
            temp_mean = np.mean(res_ps[:,group,i])
            temp_std = np.std(res_ps[:,group,i])
            plt.errorbar(x[i]+offset[group], temp_mean, yerr=temp_std, fmt = 'o', color = colors[group], elinewidth = 2)
    plt.hlines(0,4,52,colors='gray',linestyle=':')
    plt.xlabel('stim amp')
    plt.ylabel('residuals')
    plt.title(f'Error in Response Proportion Prediction: {K}-Fold Cross-Validation of {model_name} Model')
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=labels[i],
               markerfacecolor=colors[i], markeredgecolor=colors[i], markersize=8,
               linestyle='None') for i in range(4)
    ]
    x_ticks = np.array([6, 12, 18, 24, 28, 32, 38, 44, 50])
    plt.ylim(*ylims)
    plt.xticks(x_ticks)
    plt.legend(handles=legend_handles, title="Mean ± SD")
    plt.tight_layout()
    if plot_save:
        plt.savefig(f'{model_name[:5]}_{K}Fold_CV_plot.png')
    plt.show()

#%%
K=3
reps = 5
Nsess=43
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~RUNNING BAYESIAN~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#split data and do repeated bayesian predictions
total_num_models = reps*4
model_running = 0
split_data_all = {}
C_pred = {}
for rep in range(reps):
    split_data_all[rep] = {}
    C_pred[rep] = {}
    for grp in ['ld','ln','rd','rn']:
        model_running += 1
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'~~~ Fitting model set {model_running} of {total_num_models} ~~~')
        ready_data = K_fold_train_test_split(data_dict[grp]['C_mat'], N = data_dict[grp]['N_mat'], K=K)
        split_data_all[rep][grp] = ready_data
        training_sets = ready_data['sess_pool']['train']
        testing_sets = ready_data['sess_pool']['test']
        C_pred[rep][grp] = bayes_preds(training_sets, testing_sets, K=K)

#%%

#format bayesian predictions
preds = np.zeros((reps, 4, 8))
trues = np.zeros((reps, 4, 8))
Ns = np.zeros((reps, 4, 8))
# Fill it
for i in range(reps):
    for j, group in enumerate(['ld', 'ln', 'rd', 'rn']):
        preds[i, j, :] = C_pred[i][group]
        trues[i, j, :] = split_data_all[i][group]['sess_pool']['all']['C']
        Ns[i, j, :] = split_data_all[i][group]['sess_pool']['all']['N']


#%%
# do hierarchicical predictions

print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~RUNNING HIERARCHICAL~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

model_running = 0
C_pred_hier = {}
for rep in range(reps):
    C_pred_hier[rep] = {}
    for grp in ['ld','ln','rd','rn']:
        model_running += 1
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'~~~ Fitting model set {model_running} of {total_num_models} ~~~')
        ready_data = split_data_all[rep][grp]
        training_sets = ready_data['sess_strat']['train']
        testing_sets = ready_data['sess_strat']['test']
        C_pred_hier[rep][grp] = hier_preds(training_sets, testing_sets, K=K)

#%%

#format heirarchical predictions
hier_preds = np.zeros((reps, 4, Nsess, 8))
hier_trues = np.zeros((reps, 4, Nsess, 8))
Ns_hier = np.zeros((reps, 4, Nsess, 8))
# Fill it
for i in range(reps):
    for j, group in enumerate(['ld', 'ln', 'rd', 'rn']):
        hier_preds[i, j, :, :] = C_pred_hier[i][group]
        hier_trues[i, j, :, :] = split_data_all[i][group]['sess_strat']['all']['C']
        Ns_hier[i, j, :, :] = split_data_all[i][group]['sess_strat']['all']['N']




#%%
hier_preds_agg = np.sum(hier_preds, axis = 2)
hier_trues_agg = np.sum(hier_trues, axis = 2)
Ns_hier_agg = np.sum(Ns_hier, axis = 2)

#%%

#plot
plot_KFoldCV(hier_preds_agg, hier_trues_agg, Ns_hier_agg, K, 'Hierarchical', plot_save=True)
plot_KFoldCV(preds, trues, Ns, K, 'Bayesian', plot_save=True)


#%%
hier_pred_props_strat = hier_preds / Ns_hier
trues_props_strat = hier_trues / Ns_hier

bayes_pred_props_agg = preds/Ns
bayes_pred_props_strat = np.stack([bayes_pred_props_agg] * 43, axis=2)
bayes_preds_strat = bayes_pred_props_strat*Ns_hier

#%%

results_pool = {'N': Ns, 
                'C': {'true': trues,
                    'bayes': {'est': preds, 'rmses': 9},
                    'hier': {'est': hier_preds_agg, 'rmses': 9}}, 
                'P': {'true': trues/Ns,
                    'bayes': {'est': preds/Ns, 'rmses': 9},
                    'hier': {'est': hier_preds_agg/Ns, 'rmses': 9}}}
results_strat = {'N': Ns_hier, 
                'C': {'true': hier_trues,
                    'bayes': {'est': bayes_preds_strat, 'rmses': 9},
                    'hier': {'est': hier_preds, 'rmses': 9}}, 
                'P': {'true': trues_props_strat,
                    'bayes': {'est': bayes_pred_props_strat, 'rmses': 9},
                    'hier': {'est': hier_pred_props_strat, 'rmses': 9}}}

for res in [results_pool, results_strat]:
    for form in ['C', 'P']:
        temp_true = res[form]['true'].reshape(-1,8)
        for mod in ['bayes', 'hier']:
            temp_pred = res[form][mod]['est'].reshape(-1,8)
            res[form][mod]['rmses'] = mean_squared_error(temp_true, temp_pred, squared = False, multioutput = 'raw_values')

# rmse = mean_squared_error(C_true.T,C_pred.T,squared=False)


# plt.scatter(C_true.reshape((1,-1)),C_pred.reshape((1,-1))-C_true.reshape((1,-1)))
# plt.hist(C_pred.reshape(32)-C_true.reshape(32))


#%%

for res in [results_pool, results_strat]:
    for form in ['C', 'P']:
        temp_true = res[form]['true'].reshape(-1,8)
        for mod in ['bayes', 'hier']:
            temp_pred = res[form][mod]['est'].reshape(-1,8)
            temp_errs = temp_pred-temp_true
            res[form][mod]['err_means'] = np.mean(temp_errs, axis=0)
            res[form][mod]['err_stds'] = np.std(temp_errs, axis=0)
            res[form][mod]['err_mins'] = np.min(temp_errs, axis=0)
            res[form][mod]['err_maxs'] = np.max(temp_errs, axis=0)
            
            
#%%
cols = ['orange','green']
model_names = ['Bayesian', 'Hierarchical']
res_type = ['Pooled-Session', 'Session-Specific']

offsets = [-0.5,0.5]
form = 'P'

i = 1
for i, res in enumerate([results_pool, results_strat]):
    for j, mod in enumerate(['bayes', 'hier']):
        col = cols[j]
        offset = offsets[j]
        temp_means = res[form][mod]['err_means']
        temp_stds = res[form][mod]['err_stds']
        temp_mins = res[form][mod]['err_mins']
        temp_maxs = res[form][mod]['err_maxs']
        plt.errorbar(x+offset, temp_means, yerr=temp_stds, fmt = 'o', elinewidth = 3, capsize = 3, color = col, label = model_names[j])
        #plt.errorbar(x+offset, temp_means, yerr=[temp_means-temp_mins, temp_maxs-temp_means], fmt = 'o', elinewidth = 1, color = col)
        x_ticks = np.array([6, 12, 18, 24, 28, 32, 38, 44, 50])
        plt.xticks(x_ticks)
        # plt.scatter(x+offset, temp_mins, marker = '*', c=col)
        # plt.scatter(x+offset, temp_maxs, marker = '*', c=col)
        plt.legend(title="Mean ± SD")
        plt.title(res_type[i])
        plt.xlabel('Stimulus Amplitude')
        plt.ylabel('Error in Predicted Response Probability')
    plt.savefig(f'{res_type[i][:5]}_{K}Fold_errs_plot.png')
    plt.show()

#%%

for i, res in enumerate([results_pool, results_strat]):
    print(res_type[i])
    for j, mod in enumerate(['bayes', 'hier']):
        print('~~~'+model_names[j])
        for form in ['C', 'P']:
            print('~~~~~~'+form)
            print('~~~~~~'+str(np.mean(res[form][mod]['rmses'])))

            
#%%
trace_bayes = fit_bayesian_model(ready_data['sess_pool']['all']['C'], ready_data['sess_pool']['all']['N'], trace = True)
trace_hier = fit_hier_model(ready_data['sess_strat']['all']['C'], ready_data['sess_strat']['all']['N'], trace = True)

#%%

az.plot_trace(trace_bayes, var_names=['gam', 'lam', 'pse', 'jnd'], figsize=(15, 10), combined=False)

# Adjust spacing between plots
plt.tight_layout(pad=2.0)  # Increase pad as needed
plt.savefig('bayes_traceplots.png')
plt.show()

#%%
s=25
az.plot_trace(trace_hier, var_names=['gamma_h', 'gamma_l', 'PSE', 'JND'], figsize=(15, 10), coords = {'gamma_h_dim_0':[s], 'gamma_l_dim_0':[s], 'PSE_dim_0':[s], 'JND_dim_0':[s]},  combined=False)

# Adjust spacing between plots
plt.tight_layout(pad=2.0)  # Increase pad as needed
plt.savefig('hier_session_traceplots.png')
plt.show()

#%%

az.plot_pair(trace_hier, var_names=['xi'], figsize=(15, 10), coords = {'xi_dim_0':[0, 2, 4, 6]}, kind = 'kde')

# Adjust spacing between plots
plt.tight_layout(pad=2.0)  # Increase pad as needed
plt.savefig('hier_post_xi.png')
plt.show()

#%%

az.plot_pair(trace_hier, var_names=['gamma_h', 'gamma_l', 'PSE', 'JND'], figsize=(15, 10), coords = {'gamma_h_dim_0':[s], 'gamma_l_dim_0':[s], 'PSE_dim_0':[s], 'JND_dim_0':[s]}, kind = 'kde')

# Adjust spacing between plots
plt.tight_layout(pad=2.0)  # Increase pad as needed
plt.savefig('hier_post_desc.png')
plt.show()



