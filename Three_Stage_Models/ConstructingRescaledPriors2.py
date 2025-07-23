# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 17:08:49 2025

@author: schro
"""


from scipy.stats import beta as ssbeta
from scipy.stats import gamma as ssgamma



mu_xi = 0.05
sig_xi = 0.01
[L,U] = [0, 0.25]



def comp_hyps(lik, supp):
    
    [L,U] = supp
    mu_xi = np.mean(lik)
    sig_xi = (lik[1]-lik[0])/2
    
    
    mu_z = (mu_xi-L)/(U-L)
    sig_z = sig_xi/(U-L)
    
    kappa = mu_z*(1-mu_z)/(sig_z**2) - 1
    
    a = mu_z*kappa
    b = (1-mu_z)*kappa
    return [a, b]


x_orig = np.array([6, 12, 18, 24, 32, 38, 44, 50])
x_new = (x_orig - np.mean(x_orig) ) / np.std(x_orig)

#%%

supp_PSE = np.array([min(x_new),max(x_new)])
lik_PSE = np.array([x_new[2],x_new[5]])



a_PSE, b_PSE = comp_hyps(lik_PSE, supp_PSE)


#%%

supp_JND = np.array([0.04,max(x_new)])
lik_JND = np.array([0.14,0.75])

a_JND, b_JND = comp_hyps(lik_JND, supp_JND)

#%%

supp_gamma = np.array([0,0.24999])
lik_gamma = np.array([0,0.08])


a_gamma, b_gamma = comp_hyps(lik_gamma, supp_gamma)

true_ex = a_gamma/(a_gamma+b_gamma)

for i in [2,3,4,5,6,7]:
    atemp = round(a_gamma, i)
    btemp = round(b_gamma, i)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"Rounding to {i} digits:")
    ex_rnd = atemp/(atemp+btemp)
    perc_err = abs(ex_rnd-true_ex)*100/true_ex
    print(f"Estimated expectation = {ex_rnd}")
    err_val =  abs(ex_rnd-true_ex)
    print(f"Error = {err_val}")
    print(f"Percent Error = {perc_err} % ")
    
#%%

hyper_prior_df = pd.DataFrame.from_dict(
    {'PSE': {'supp': np.round(supp_PSE, 4), 'hparams' : np.round(np.array([a_PSE,b_PSE]), 4) }, 
     'JND':  {'supp': np.round(supp_JND, 4), 'hparams' : np.round(np.array([a_JND,b_JND]), 4) }, 
     'gamma': {'supp': np.round(supp_gamma, 4), 'hparams' : np.round(np.array([a_gamma,b_gamma]), 4) }
     }, orient='index')


with open('desc_hyper_priors.pkl', 'wb') as f:
    pickle.dump(hyper_prior_df, f)
