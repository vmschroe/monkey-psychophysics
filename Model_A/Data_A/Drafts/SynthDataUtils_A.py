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
    
