# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 16:07:34 2025

@author: schro
"""

###############################################
## Import packages and specify some settings ##
###############################################
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import norm

# This makes plots show up and look nice
sns.set(context='paper',style='white',font_scale=1.5,rc={"lines.linewidth":2.5})
sns.set_palette('muted')

###############################################
###############################################


#%% Problem 1 part 1

def firing_rate_EIF(Ix, plot_spikes = False, T=1000, dt=.1, EL=-72, taum=15, Vth=5, Vre=-75, VT=-55, D=2, V0=-70, plot_title = 'EIF'):
    
    # Discretized time
    time=np.arange(0,T,dt)
    
    # Applied current
    if isinstance(Ix, (int, float, np.float64)):
        Ix = np.full(time.shape, Ix, dtype=float)
    
    # Compute V using the forward Euler method
    V=np.zeros_like(time)
    SpikeTimes=np.array([])
    V[0]=V0
    for i in range(len(time)-1):
        # Euler step
        V[i+1]=V[i]+dt*(-(V[i]-EL)+D*np.exp((V[i]-VT)/D)+Ix[i])/taum
    
        # Threshold-reset condition
        if V[i+1]>=Vth:
            V[i+1]=Vre
            V[i]=Vth  # This makes plots nicer
            SpikeTimes=np.append(SpikeTimes,time[i+1])
            
    if plot_spikes:
        # Make figure
        plt.subplots(1,2,figsize=(8, 2.5))

        plt.subplot(1,2,1)
        plt.plot(time,Ix,color='r')
        plt.xlabel('time (ms)')
        plt.ylabel('I$_x$ (mV)')
        sns.despine()


        plt.subplot(1,2,2)
        plt.plot(time,V,color='gray')
        plt.plot(SpikeTimes,Vth+1+0*SpikeTimes,'ro')
        plt.xlabel('time (ms)')
        plt.ylabel('V (mV)')
        sns.despine()
        plt.tight_layout()
        plt.suptitle(plot_title)
        
    r = len(SpikeTimes)/T
    return r


I0s = np.arange(0,30,0.2)
rs = []
for I in I0s:
    r = firing_rate_EIF(I)
    rs.append(r)

rs = np.array(rs)
plt.plot(I0s,rs)
plt.xlabel('Input I$_0$ (mV)')
plt.ylabel('Firing Rate r (kHz)')
plt.title('f-I Curve: EIF model, time-constant input')

rs_EIF = rs

#%% Problem 1 part 2

def firing_rate_LIF(Ix, plot_spikes = False, T=1000, dt=.1, EL=-72, taum=15, Vth=5, Vre=-75, VT=-55, D=2, V0=-70):
    
    # Discretized time
    time=np.arange(0,T,dt)
    
    # Applied current
    if isinstance(Ix, (int, float, np.float64)):
        Ix = np.full(time.shape, Ix, dtype=float)
    
    # Compute V using the forward Euler method
    V=np.zeros_like(time)
    SpikeTimes=np.array([])
    V[0]=V0
    for i in range(len(time)-1):
        # Euler step
        V[i+1]=V[i]+dt*(-(V[i]-EL)+Ix[i])/taum
    
        # Threshold-reset condition
        if V[i+1]>=VT:
            V[i+1]=Vre
            V[i]=VT  # This makes plots nicer
            SpikeTimes=np.append(SpikeTimes,time[i+1])
            
    if plot_spikes:
        # Make figure
        plt.subplots(1,2,figsize=(8, 2.5))

        plt.subplot(1,2,1)
        plt.plot(time,Ix,color='r')
        plt.xlabel('time (ms)')
        plt.ylabel('I$_x$ (mV)')
        plt.title('A',loc='left')
        sns.despine()


        plt.subplot(1,2,2)
        plt.plot(time,V,color='gray')
        plt.plot(SpikeTimes,VT+1+0*SpikeTimes,'ro')
        plt.xlabel('time (ms)')
        plt.ylabel('V (mV)')
        sns.despine()
        plt.title('B',loc='left')
        plt.tight_layout()
        
    r = len(SpikeTimes)/T
    return r


#firing_rate_LIF(18,plot_spikes = True)



rs = []
for I in I0s:
    r = firing_rate_LIF(I)
    rs.append(r)

rs = np.array(rs)
plt.plot(I0s,rs)
plt.xlabel('Input I$_0$ (mV)')
plt.ylabel('Firing Rate r (kHz)')
plt.title('f-I Curve: LIF model, time-constant input')

rs_LIF = rs

#%% Problem 1 part 3


def firing_rate_HH(Ix, plot_spikes = False, T=400, dt=.001):

    # Discretized time
    
    time=np.arange(0,T,dt)
    
    # Applied current
    if isinstance(Ix, (int, float, np.float64)):
        Ix = np.full(time.shape, Ix, dtype=float)
    
        StepTime=20
        StepWidth=10
        Ix=Ix*norm.cdf(time,StepTime,StepWidth)
    
    # Define gating variables as inline functions
    alphan = lambda V: .01*(V+55)/(1-np.exp(-.1*(V+55)))
    betan = lambda V: .125*np.exp(-.0125*(V+65))
    alpham = lambda V: .1*(V+40)/(1-np.exp(-.1*(V+40)))
    betam = lambda V: 4*np.exp(-.0556*(V+65))
    alphah = lambda V: .07*np.exp(-.05*(V+65))
    betah = lambda V: 1/(1+np.exp(-.1*(V+35)))
    
    
    # n variable
    ninfty= lambda V: (alphan(V)/(alphan(V)+betan(V)))
    taun= lambda V: (1/(alphan(V)+betan(V)))
    minfty= lambda V: (alpham(V)/(alpham(V)+betam(V)))
    taum= lambda V: (1/(alpham(V)+betam(V)))
    hinfty= lambda V: (alphah(V)/(alphah(V)+betah(V)))
    tauh= lambda V: (1/(alphah(V)+betah(V)))
    
    # Parameters
    Cm=1
    gL=.3
    EL=-54.387
    gK=36
    EK=-77
    gNa=120
    ENa=50
    
    # Initial conditions near their fixed points when Ix=0
    V0=-65.0
    n0=ninfty(V0)
    m0=minfty(V0)
    h0=hinfty(V0)
    
    
    # Currents
    IL= lambda V: (-gL*(V-EL))
    IK = lambda n,V: (-gK*n **4*(V-EK))
    INa = lambda m,h,V: (-gNa*m **3*h*(V-ENa))
    
    # Toal ion currents
    Iion = lambda n,m,h,V: IL(V)+IK(n,V)+INa(m,h,V)
    
    # Euler solver
    V=np.zeros_like(time)
    n=np.zeros_like(time)
    m=np.zeros_like(time)
    h=np.zeros_like(time)
    V[0]=V0
    n[0]=n0
    m[0]=m0
    h[0]=h0
    for i in range(len(time)-1):
        # Update gating variables
        n[i+1]=n[i]+dt*((1-n[i])*alphan(V[i])-n[i]*betan(V[i]))
        m[i+1]=m[i]+dt*((1-m[i])*alpham(V[i])-m[i]*betam(V[i]))
        h[i+1]=h[i]+dt*((1-h[i])*alphah(V[i])-h[i]*betah(V[i]))
    
        # Update membrane potential
        V[i+1]=V[i]+dt*(Iion(n[i],m[i],h[i],V[i])+Ix[i])/Cm
    
    time_chopped = time[time>=100]
    V_chopped = V[time>=100]
    peaks, _ = find_peaks(V_chopped, height=0)
    r = len(peaks)/(T-100)
    
    if plot_spikes:
        # Make figure
        plt.subplots(1,2,figsize=(11,2.5))
        
        plt.subplot(1,2,1)
        plt.plot(time,Ix,'m')
        plt.ylabel(r'$I_{x}$')
        plt.xlabel('time (ms)')
        plt.title('A',loc='left')
        sns.despine()
        
        
        plt.subplot(1,2,2)
        plt.plot(time,V,color=sns.color_palette()[7])
        plt.ylabel('V (mV)')
        plt.xlabel('time (ms)')
        plt.title('B',loc='left')
        plt.axis([100,T,np.min((-70,np.min(V))),np.max((-57,np.max(V)))])
        sns.despine()
        
        plt.tight_layout()
    return r


rs_HH = []
for I in I0s:
    r = firing_rate_HH(I)
    rs_HH.append(r)

rs_HH = np.array(rs_HH)
plt.plot(I0s,rs_HH)
plt.xlabel('Input I$_0$ (mV)')
plt.ylabel('Firing Rate r (kHz)')
plt.title('f-I Curve: HH model, time-constant input')


#%% Problem 1 Summary

plt.plot(I0s,rs_LIF, label = "LIF Model", alpha=0.75)
plt.plot(I0s,rs_EIF, label = "EIF Model", alpha=0.75)
plt.plot(I0s,rs_HH, label = "HH Model", alpha=0.75)
plt.xlabel('Input I$_0$ (mV)')
plt.ylabel('Firing Rate r (kHz)')
plt.legend()
plt.title('f-I Curve Comparison')


#%% Problem 2

T=1000
dt=.1
    
# Discretized time
time=np.arange(0,T,dt)

A = 290

omega = 100/100

Ix_sin = A* np.sin(omega*time)

plot_title = f"EIF, A = {A}, $\omega$ = {omega}"

r_sin = firing_rate_EIF(Ix_sin, plot_spikes = True, T=1000, dt=.1, plot_title = plot_title)


#%% Problem 3

###############################################
## Import packages and specify some settings ##
###############################################


# This makes plots show up and look nice
sns.set(context='paper',style='white',font_scale=1.5,rc={"lines.linewidth":2.5})
sns.set_palette('muted')


###############################################
###############################################

# Discretized time
T=300
dt=.1 
time=np.arange(0,T,dt)


# Synapse parameters
taue=5 
Je=12
taui=5
Ji=-12

# Neuron parameters
EL=-72 
taum=10


# Presynaptic spike times
ExcSpikeTimes=np.array([20,200,210])
InhSpikeTimes=np.array([100])

# Binarized presynaptic spike train
Se=np.zeros_like(time)
Si=np.zeros_like(time)
Se[np.floor(ExcSpikeTimes/dt).astype(int)]=1/dt
Si[np.floor(InhSpikeTimes/dt).astype(int)]=1/dt

# External input. Set to zero.
I0=0
Ix=I0+np.zeros_like(time)


# Euler solver to compute Is and V
Ie=np.zeros_like(time)
Ii=np.zeros_like(time)
V_delta=np.zeros_like(time)
V_delta[0]=EL 
for i in range(len(time)-1):
    V_delta[i+1]=V_delta[i]+dt*(-(V_delta[i]-EL)+Je*Se[i]+Ji*Si[i]+Ix[i])/taum 



Ie=np.zeros_like(time)
Ii=np.zeros_like(time)
V_exp=np.zeros_like(time)
V_exp[0]=EL 
for i in range(len(time)-1):
    V_exp[i+1]=V_exp[i]+dt*(-(V_exp[i]-EL)+Ie[i]+Ii[i]+Ix[i])/taum 
    Ie[i+1]=Ie[i]+dt*(-Ie[i]+Je*Se[i])/taue
    Ii[i+1]=Ii[i]+dt*(-Ii[i]+Ji*Si[i])/taui


# Make figure
plt.plot(time,V_delta,label = 'delta synapses')
plt.plot(time,V_exp,label = 'exponential PSE')
plt.xlabel('time (ms)')
plt.ylabel('V (mV)')
sns.despine()
plt.title('Membrane Potential Traces')
plt.legend()
plt.show()