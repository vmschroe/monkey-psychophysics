#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick check for periodicity in the session PSEs, with each amplitude's mean level removed:
Lomb-Scargle periodogram on calendar days and lag-1 autocorrelation against sessions shuffled within
amplitude, a semivariogram, left-right correlations and a weekday comparison. Weekdays assume the
calendar of 2000, because the data have no year.

Uses the PSE draws cached by PSE_Within_Amplitude_B.py. Set DROP_SESSION to a date, such as '04-10',
to repeat the check without that session.
"""

from pathlib import Path
import numpy as np, pandas as pd, pickle, warnings
from scipy.signal import lombscargle
warnings.filterwarnings('ignore')
EXPLORATORY_DIR=Path(__file__).resolve().parent
MODEL_DIR=EXPLORATORY_DIR.parent
DROP_SESSION=None
d=pickle.load(open(MODEL_DIR/'Data_B'/'ReadyData_Sirius_B.pkl','rb'))
p=pickle.load(open(EXPLORATORY_DIR/'PSE_Draws_B.pkl','rb'))
dates=[str(s) for s in p.sessions.values]; s=d['session_summary']; s.index=s.index.astype(str)
amp=s['dist_amp'].reindex(dates).values.astype(float)
day=(pd.to_datetime(['2000-'+x for x in dates])-pd.Timestamp('2000-01-01')).days.values.astype(float)
m={g:(p.sel(groups=g).mean(('chain','draw')).values*14.6287+28) for g in p.groups.values}
sd={g:(p.sel(groups=g).std(('chain','draw')).values*14.6287) for g in p.groups.values}
use=np.array([(amp==a).sum()>=2 for a in amp])
def resid(y,a):
    return y-pd.Series(y).groupby(a).transform('mean').values
raw={'L uni':m['left_uni'],'L bi':m['left_bi'],'R uni':m['right_uni'],'R bi':m['right_bi']}
raw['L shift']=raw['L bi']-raw['L uni']; raw['R shift']=raw['R bi']-raw['R uni']
raw['mean of 4']=sum(m.values())/4
if DROP_SESSION: use[dates.index(DROP_SESSION)]=False
t,a=day[use],amp[use]
periods=np.linspace(2.5,60,400); w=2*np.pi/periods
rng=np.random.default_rng(1)
def peak(y):
    P=lombscargle(t,y-y.mean(),w,normalize=True); i=P.argmax(); return P[i],periods[i],P
def lag1(y):
    return np.corrcoef(y[:-1],y[1:])[0,1]
def shuffle(y):
    z=y.copy()
    for v in np.unique(a):
        i=np.flatnonzero(a==v); z[i]=rng.permutation(z[i])
    return z
rows=[]
for k,y0 in raw.items():
    y=resid(y0[use],a)
    pk,per,P=peak(y); l=lag1(y)
    nul=[shuffle(y) for _ in range(4000)]
    np_=np.array([peak(z)[0] for z in nul]); nl=np.array([lag1(z) for z in nul])
    rows.append(dict(series=k,sd_resid=y.std(),peak_period_days=per,peak_power=pk,p_peak=(np_>=pk).mean(),lag1=l,p_lag1_pos=(nl>=l).mean()))
print(pd.DataFrame(rows).round(3).to_string())
print('typical session posterior sd:',{g:round(float(np.median(v)),2) for g,v in sd.items()})
ry={k:resid(v[use],a) for k,v in raw.items()}
print('left-right residual r: uni %.2f bi %.2f'%(np.corrcoef(ry['L uni'],ry['R uni'])[0,1],np.corrcoef(ry['L bi'],ry['R bi'])[0,1]))
# variogram on real day gaps, mean of 4
y=ry['mean of 4']; i,j=np.triu_indices(len(y),1); g=t[j]-t[i]; sq=(y[i]-y[j])**2/2
bins=[(1,1),(2,4),(5,7),(8,10),(11,14),(15,21),(22,28),(29,35),(36,50),(51,70),(71,124)]
print('semivariogram, mean of 4:',[(b,int(((g>=b[0])&(g<=b[1])).sum()),round(float(sq[(g>=b[0])&(g<=b[1])].mean()),1)) for b in bins]); print('variance',round(float(y.var()),1))
dow=pd.to_datetime(['2000-'+x for x in dates]).dayofweek.values[use]
print('day-of-week n, mean residual (mean of 4), Mon=0:',pd.Series(y).groupby(dow).agg(['size','mean']).round(2).to_dict('index'))
from scipy.stats import f_oneway
print('dow anova p (assumes 2000 weekdays, year unknown, for the gap pattern only):',round(f_oneway(*[y[dow==k] for k in np.unique(dow) if (dow==k).sum()>1]).pvalue,3))
