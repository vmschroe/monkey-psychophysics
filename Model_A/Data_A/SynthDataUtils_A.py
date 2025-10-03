# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:24:57 2025

@author: schro
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from scipy.stats import binom
import pickle
from scipy.stats import gamma
from scipy.stats import beta
import ast
from scipy.stats import truncnorm


with open("Data_Frame_A.pkl", "rb") as f:
    data = pickle.load(f)  
    
def synth_generator_A(beta0_fix, beta1_fix, gam_h_fix, gam_l_fix):
    data['response'] = 0
    data['psi'] = gam_h_fix[ data['side_idx'] , data['dist_idx'] ] + (1- gam_h_fix[ data['side_idx'] , data['dist_idx'] ] - gam_l_fix[ data['side_idx'] , data['dist_idx'] ])/ (1 + np.exp(-(beta0_fix[ data['side_idx'] , data['dist_idx'] ] + beta1_fix[ data['side_idx'] , data['dist_idx'] ] * data['stim'])))
    data['response'] = binom.rvs(n=1, p=np.array(data['psi']))
    return data
    