# -*- coding: utf-8 -*-
"""
Created on Wed May 14 18:30:21 2025

@author: schro
"""

import FunctionsForGeneralized as ffg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = [6, 12, 18, 24, 32, 38, 44, 50]

#ffg.phi_L(params,X)

params = [0.09, 0.06, -6.7, 0.209]

x_vals = np.linspace(0,56,num=200)
y_vals = ffg.phi_L(params,x_vals)
x25 = ffg.phi_inv_L(params, 0.25)
x50 = ffg.phi_inv_L(params, 0.5)
x75 = ffg.phi_inv_L(params, 0.75)



plt.plot(x_vals,y_vals,'green', label=r'$\Psi(x_\text{amp})$')
plt.hlines(params[0], 0,56, colors = 'blue', linestyles = 'dashed', label = r'low-amplitude lapse rate $\gamma_l$')
plt.hlines(1-params[1], 0,56, colors = 'purple', linestyles = 'dashed', label = r'1-(high-amplitude lapse rate $\gamma_h$)')
plt.hlines([0.25,0.5,0.75], 0,[x25,x50,x75], colors = ['orange','red','orange'], linestyles = 'dotted')

plt.vlines(28,0,1,colors='gray',linestyles = 'dotted', label='threshold')
plt.vlines([x25,x50,x75],0,[0.25,0.5,0.75],colors = ['orange','red','orange'], linestyles = 'dotted')

plt.xlabel(r'Amplitude of stimulus $x_\text{amp}$',fontsize=12)
plt.ylabel('Prob[choice = "high"]',fontsize=12)
plt.title(r'Psychometric function $\Psi(x_\text{amp})$',fontsize=14)
plt.axis([0, 56, 0, 1])
plt.xticks([6, 12, 18, 24, 28, 32, 38, 44, 50])
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.savefig('ExamplePsychFunc.png',dpi=300)
plt.show()