# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 13:35:04 2025

@author: schro
"""

import datetime
import pickle
import pandas as pd
import os

with open("Data/session_summary.pkl", "rb") as f:
    sess_sum = pickle.load(f)  

#add dates to sess_sum
sess_dates = []
for sess_name in sess_sum.index.tolist():
    sess_dates.append(datetime.date(2023, int(sess_name[0:2]), int(sess_name[3:5])))
    
sess_sum['dates'] = sess_dates

#add weekdays to sess_sum, 0 = monday

weekdays = []
for sess_date in sess_sum['dates'].tolist():
    weekdays.append(sess_date.weekday())
    
sess_sum['weekday'] = weekdays

#add juice vs water
rewards = []
dates = []
sessions = []
directory_path = 'Data/Sirius_data'
for filename in os.listdir(directory_path):
    if filename.endswith('.xlsx'):
        # Construct the full file path
        file_path = os.path.join(directory_path, filename)
        df = pd.read_excel(file_path, engine='openpyxl')
        sess_date = filename[8:13]
        list.append(dates, sess_date)
        sess_num = dates.count(sess_date)
        session = sess_date + "-S" + str(sess_num)
        list.append(sessions,session)
        rewardlist = df['rewardType'].unique()
        if ('juice' in rewardlist) or ('Juice' in rewardlist):
            reward = 'juice'
        elif ('water' in rewardlist) or ('Water' in rewardlist):
            reward = 'water'
        else:
            reward = 'check'
            print(filename)
        list.append(rewards,reward)

rewdf = pd.DataFrame()
rewdf['Session'] = sessions
rewdf['reward'] = rewards
rewdf = rewdf.set_index('Session')
sess_sum = sess_sum.join(rewdf)



#posterior estimates for each descriptor
descriptors = ['PSE','JND','gamma_h','gamma_l']
post_ests = {}
for grp in grps:
    post_ests[grp] = {}
    for desc in descriptors:
        post_ests[grp][desc] = {}
        post_ests[grp][desc]['mean'] = np.array(az.summary(traces[grp], var_names = desc)['mean'])
        post_ests[grp][desc]['lHDI'] = np.array(az.summary(traces[grp], var_names = desc)['hdi_3%'])
        post_ests[grp][desc]['hHDI'] = np.array(az.summary(traces[grp], var_names = desc)['hdi_97%'])

with open('H3sL_descrip_post_ests.pkl', 'wb') as file:
     pickle.dump(post_ests, file)


import matplotlib.pyplot as plt

desc = 'gamma_h'
x_vals = sess_sum['dates']
numtrials = sess_sum['TrialsTotal']
rewardtype = sess_sum['reward']
dist = sess_sum['Dist_AMP']

fig, axs = plt.subplots(3, 1, sharex=True)
# Remove vertical space between Axes
fig.subplots_adjust(hspace=0)

# Plot each graph, and manually set the y tick values
axs[0].plot(x_vals, dist)

axs[1].plot(x_vals, post_ests['ln'][desc]['mean'], color = 'blue')
axs[1].plot(x_vals, post_ests['ld'][desc]['mean'], color = 'red')

axs[2].plot(x_vals, post_ests['ln'][desc]['mean'], color = 'blue')
axs[2].plot(x_vals, post_ests['ld'][desc]['mean'], color = 'red')

plt.show()