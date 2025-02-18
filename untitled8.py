#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:25:08 2024

@author: vmschroe
"""
# is it working?

import numpy as np
import matplotlib.pyplot as plt 
# Consider a family of explicit Adams-Bashforth multistep methods - Linear Multistep Methods


# Write class Integrator that will use AB2 (2-step) and AB3 (3-step) methods.

# Use RK2 as a startup method - RK Methods


 

# Pass the following parameters (in this order) during the initialization -

# integ = Integrator(ABk, N, dt, IC, f)

class Integrator:
    
    def __init__(self, ABk, N, dt, IC, f):
        # ABk = "AB2" or "AB3" determines which method will be used
        # N = Dimension of the problem
        # dt = time-step
        # IC \in R^N numpy array of initial conditions
        # f = f(x) function which accepts x \in R^N (numpy array) as argument, and returns the right-hand side rhs \in R^N (numpy array)
        # Function f(x) has to be defined outside of the class and determines which equation we're solving.
        self.ABk = ABk  # Assign ABk to an instance variable
        self.dt = dt    # You also need to assign dt to be used in the steps
        self.f = f      # Assign the function f to be used later
        self.x = np.array(IC).reshape(1, -1)  # Ensure self.x is a 2D array
        self.fx = f(self.x[0])  # Pass only the first row (a 1D array) to f
        self.fx = self.fx.reshape(1, -1)  # Ensure self.fx is also 2D
        self.ii = 0
    def rk2_step(self):
        ii = self.ii
        dt = self.dt
        xn = self.x[ii,:]
        fxn = self.fx[ii,:]
        esthalf = xn + 0.5*dt* fxn
        xnp1 = xn + dt* self.f(esthalf)
        fxnp1 = self.f(xnp1)
        self.x = np.vstack((self.x,xnp1))
        self.fx= np.vstack((self.fx,fxnp1))
        self.ii += 1
    def AB2_step(self):
        n=self.ii-2
        dt = self.dt
        xn = self.x[n,:]
        xnp1 = self.x[n+1,:]
        fxn = self.fx[n,:]
        fxnp1 = self.fx[n+1,:]
        #x_{n+2} = x_{n+1} + ( 1.5* f(x_{n+1}) - 0.5* f(x_n))*dt
        xnp2 = xnp1 + (1.5*fxnp1 - 0.5*fxn)*dt
        fxnp2 = self.f(xnp2)
        self.x = np.vstack((self.x,xnp2))
        self.fx= np.vstack((self.fx,fxnp2))
        self.ii += 1
    def AB3_step(self):
        n=self.ii-3
        dt = self.dt
        xn = self.x[n,:]
        xnp1 = self.x[n+1,:]
        xnp2 = self.x[n+2,:]
        fxn = self.fx[n,:]
        fxnp1 = self.fx[n+1,:]
        fxnp2 = self.fx[n+2,:]
        #x_{n+3} = x_{n+2} + ( (23/12) * f(x_{n+2}) - (16/12) * f(x_{n+1}) + (5/12) * f(x_{n})  )*dt
        xnp3 = xnp2 + ( (23/12) * fxnp2 - (16/12) * fxnp1 + (5/12) * fxn  )*dt
        fxnp3 = self.f(xnp3)
        self.x = np.vstack((self.x,xnp3))
        self.fx= np.vstack((self.fx,fxnp3))
        self.ii += 1
    def make_one_step(self):
        ii=self.ii
        if self.ABk == 'AB2':
            if ii==0:
                X.rk2_step()
                return
            #one step of AB2
            X.AB2_step()
            return
        elif self.ABk == 'AB3':
            if ii==0 or ii==1:
                X.rk2_step()
                return
            #one step of AB3
            X.AB3_step()
        else:
            print('Error: method not recognized')

# Test the integrator class on the 
params = {
    "dt": 0.01, # time step of integration
    "sigma": 10.,
    "beta": 8/3.,
    "rho": 28.
    }

dt = params["dt"]
sig = params["sigma"]
bet = params["beta"]
rho = params["rho"]
init = [5.,0.,1.]

def lor(xyz):
    [x,y,z] = xyz
    dxdt = sig*(y-x)
    dydt = x*(rho-z)-y
    dzdt = x*y-bet*z
    return np.array([dxdt,dydt,dzdt])

X = Integrator('AB2', 3, dt, init, lor)

for count in range(100):
    X.make_one_step()

plt.figure()
plt.plot(X.x[:,0].reshape(1, -1), X.x[:,1].reshape(1, -1), 'b-')
plt.show()