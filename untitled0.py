# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 19:12:49 2025

@author: schro
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Set seed for reproducibility
np.random.seed(42)

# 1. Simple Sinusoidal Relationship (Lag of 5 units)
t = np.arange(0, 100, 1)  # Time indices
values1 = np.sin(0.1 * t)  # Original sine wave
values2 = np.roll(values1, shift=5)  # Shifted version with lag
values2[:5] = 0  # Set first 5 values to 0 after shift

# 2. Random Noise (No correlation)
noise1 = np.random.randn(100)
noise2 = np.random.randn(100)

# 3. Boolean Series (Binary Signals with Correlation)
temp = (np.sin(0.1 * t)+noise1 > 0).astype(int)  # Converts sine wave to binary
values2 = (np.roll(temp, shift=3)+noise2 > 0).astype(int) # Shift by 3 steps

A = np.correlate(values1,values2,'full')
plt.plot(np.arange(len(A)),A)
# plt.plot(t, values1, t, values2)

lag = len(values1)-np.argmax(A)-1

tshift = np.arange(lag, len(t)+lag, 1)
plt.plot(tshift, values1, t, values2)