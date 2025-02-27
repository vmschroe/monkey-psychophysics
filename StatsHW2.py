# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:20:56 2025

@author: schro
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.integrate import quad

np.random.seed(1234)

print("-----------------------------------------------------")
print("Problem #1:")

def w_u(x):
    return np.exp(-(x**2)/2) * np.pi * (1+x**2)

### Sample from cauchy distribution:
    
n=100000
X = np.random.standard_cauchy(size=n)

### 1_(4,inf) (X)

indicator = np.float64(X>4)

### Compute approximation:
p_approx = sum(indicator * w_u(X)) / sum(w_u(X))

print("For X ~ N(0,1), compute P(X>4):")
print("Approximation using Cauchy: ", p_approx)
print("Computed using normal CDF:", 1-norm.cdf(4))



### ----------------------------------------------------------------------

print("-----------------------------------------------------")
print("Problem #2:")



def H_inv(x):
    return -(2/3) * np.log(1-x)

def f_tilde(x):
    return np.exp(-(1+x)**1.5)

def g_tilde(x):
    return np.exp( -( 1 + 1.5*x ) )

M = 1
fsamples = []
Xs = []

### Take samples:

while len(fsamples)<n:
    U1 = np.random.uniform(0, 1-np.exp(-3/2))
    
    X = H_inv(U1)
    Xs = np.append(Xs, X)
    U2 = np.random.uniform(0,1)
    
    if U2 <= f_tilde(X) / (M * g_tilde(X)):
        fsamples = np.append(fsamples, X)
        
### Compute acceptance rate:

print("Approximated acceptance rate:", len(fsamples) / len(Xs))


### Plot samples and f density:

C_inv, _ = quad(f_tilde, 0, 1)
xplot = np.linspace(-0.25,1.25,300)
fx = (1/C_inv) * f_tilde(xplot) * np.float64(xplot>0) * np.float64(xplot<1)

plt.hist(fsamples,20,density = True, label = "100000 Samples from f")

plt.plot(xplot,fx,'r', linewidth=2, label = "Normalized density f")
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Density")
plt.xlim(-0.25,1.25)
plt.legend()

plt.show()

