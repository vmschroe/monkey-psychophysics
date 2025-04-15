#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:11:56 2025

@author: vmschroe
"""


import scipy.stats
import numpy as np
import pymc as pm
import arviz as az
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import interact
import pandas as pd
import os
import math
from scipy.stats import binom

import sys
import FunctionsForGeneralized as ffg
import seaborn as sns
import pickle
import tqdm
import time
import sys
from datetime import datetime
import subprocess

np.random.seed(12345)
# Simulating for higherarchical weibul

# params = gam, lam, b0, b1
#### NOTE: For a GAMMA distribution Gamma[alpha,beta]:
    #       alpha = shape, beta = rate
    #       scipy.stats.gamma : a = alpha = shape, scale = 1/beta = 1/rate
    #       pm.Gamma : alpha = shape, beta = rate
    
#### NOTE: For a BETA distribution Beta[alpha,beta]:
    #       alpha, beta are both shape parameters
    #       scipy.stats.beta : a = alpha, b = beta
    #       pm.Beta : alpha, beta
    
    
params_fixed = np.array([0.01, 0.05, -2.8, 0.1])
hyper_fixed = np.array([[1,1],[1,1],[1,1],[3.5,30]])
params_prior_params = np.array([[2, 5], [2, 5], [1.06, 0.13], [1.06, 3.08]])
params_prior_scale = np.array([0.25, 0.25, -1, 1])
hyper_dict = {
    'gam': {
        'distribution': 'beta',
        'C': 0.25,
        'hps': {
            'alpha_prior': {
                'W_dist': 'gamma',
                'params': [1.5,0.5]
            },
            'rate_prior': {
                'dist': 'gamma',
                'params': [2,0.2]
            }
        }
    },
    'lam': {
        'distribution': 'beta',
        'C': 0.25,
        'hps': {
            'alpha_prior': {
                'W_dist': 'gamma',
                'params': [1.5,0.5]
            },
            'rate_prior': {
                'dist': 'gamma',
                'params': [2,0.2]
            }
        }
    },
    'b0': {
        'distribution': 'gamma',
        'C': -1,
        'hps': {
            'shape_prior': {
                'W_dist': 'gamma',
                'params': [1.05,0.2]
            },
            'rate_prior': {
                'dist': 'gamma',
                'params': [1.1,4.6]
            }
        }
    },
    'b1': {
        'distribution': 'gamma',
        'C': 1,
        'hps': {
            'shape_prior': {
                'W_dist': 'gamma',
                'params': [1.05,1.6]
            },
            'rate_prior': {
                'dist': 'gamma',
                'params': [11,3.4]
            }
        }
    }
}

#first just try est b1