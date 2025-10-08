# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 13:57:08 2025

@author: schro
"""

def psychfunc(params, X):
    """
    Psychometric function with lapses

    Parameters:
    params : [gamma, lambda_, beta0, beta1]
    X : Stimulus amplitude level

    Returns:
    probability of guess "high"

    """
    X = np.asarray(X)
    gam_h, gam_l, beta0, beta1 = params
    logistic = 1 / (1 + np.exp(-(beta0 + beta1 * X)))
    return gam_h + (1 - gam_h - gam_l) * logistic



param_samps = trace.posterior[['beta_vec', 'gam_h', 'gam_l']]

gam_h_samps = {}
gam_l_samps = {}
beta_0_samps = {}
beta_1_samps = {}


for grp in ["left_bi","left_uni","right_bi","right_uni"]:
    gam_h_samps[grp] = param_samps['gam_h'].sel(groups = grp).values.flatten()
    gam_l_samps[grp] = param_samps['gam_l'].sel(groups = grp).values.flatten()
    beta_0_samps[grp] = param_samps['beta_vec'].sel(groups = grp, betas='b0').values.flatten()
    beta_1_samps[grp] = param_samps['beta_vec'].sel(groups = grp, betas='b1').values.flatten()


freq_df = pd.DataFrame({'stim': cov_mat[:,1], 'grp_idx': grp_idx, 'obs_data': obs_data})
freqs = pd.pivot_table(
    freq_df, 
    values='obs_data',
    index='stim',
    columns='grp_idx',
    aggfunc='mean'
)

xfit = np.linspace(-1.6,1.6,500)
y_samples = {}
hdis = {}
rec_params = {}
yrec = {}

for grp_i, grp in enumerate(['left_uni','left_bi','right_uni','right_bi']):
    y_samples[grp] = np.array([psychfunc([gam_h,gam_l,beta_0,beta_1], xfit) 
                          for gam_h,gam_l,beta_0,beta_1 in zip(
                              gam_h_samps[grp], gam_l_samps[grp], beta_0_samps[grp], beta_1_samps[grp])])
    hdis[grp] = az.hdi(y_samples[grp], hdi_prob=0.95)
    rec_params[grp] = np.mean(np.array([gam_h_samps[grp], gam_l_samps[grp], beta_0_samps[grp], beta_1_samps[grp]]), axis = 1)
    
    yrec[grp] = psychfunc(rec_params[grp], xfit)
    
    





#######%%
#######

x_old = np.array([6, 12, 18, 24, 32, 38, 44, 50])
x_mu = np.mean(x_old)
x_sig = np.std(x_old)
stims_normed = ((x_old-x_mu)/x_sig)

xfit2 = x_mu+x_sig*xfit



fig, axs = plt.subplots(1, 2, figsize=(10, 4)) # figsize adjusts the figure size

# Plot on the first subplot


axs[0].plot(x_mu+x_sig*xfit,yrec['left_uni'],label='Unimanual Curve',color='blue')
axs[0].fill_between(x_mu+x_sig*xfit, hdis['left_uni'][:, 0], hdis['left_uni'][:, 1], color='blue', alpha=0.3, label='95% HDI')
axs[0].scatter(x_mu+x_sig*np.array(freqs.index),np.array(freqs[0]),label='Unimanual Data', color = 'blue')

axs[0].plot(x_mu+x_sig*xfit,yrec['left_bi'],label='Bimanual Curve',color='red')
axs[0].fill_between(x_mu+x_sig*xfit, hdis['left_bi'][:, 0], hdis['left_bi'][:, 1], color='red', alpha=0.3, label='95% HDI')
axs[0].scatter(x_mu+x_sig*np.array(freqs.index),np.array(freqs[1]),label='Bimanual Data', color = 'red')

axs[0].set_xlabel('Stimulus Amplitude')

axs[1].plot(x_mu+x_sig*xfit,yrec['right_uni'],label='Unimanual Curve',color='blue')
axs[1].fill_between(x_mu+x_sig*xfit, hdis['right_uni'][:, 0], hdis['right_uni'][:, 1], color='blue', alpha=0.3, label='95% HDI')
axs[1].scatter(x_mu+x_sig*np.array(freqs.index),np.array(freqs[2]),label='Unimanual Data', color = 'blue')

axs[1].plot(x_mu+x_sig*xfit,yrec['right_bi'],label='Bimanual Curve',color='red')
axs[1].fill_between(x_mu+x_sig*xfit, hdis['right_bi'][:, 0], hdis['right_bi'][:, 1], color='red', alpha=0.3, label='95% HDI')
axs[1].scatter(x_mu+x_sig*np.array(freqs.index),np.array(freqs[3]),label='Bimanual Data', color = 'red')

axs[1].set_xlabel('Stimulus Amplitude')


axs[0].legend(loc='upper left')
axs[1].legend(loc='upper left')
fig.suptitle('Psychometric Curves')

plt.tight_layout()
# Display the figure
plt.show()




