#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from scipy.stats import zscore

plt.style.use('seaborn')
plt.rcParams['font.size'] = '7'

data = pd.read_csv('/Users/zachz/Documents/timescales_analysis/1000iter results/isi figs/spiking_stats/spike_stats.csv')

data['z_n_spikes'] = data.groupby(['species','brain_region']).n_spikes.transform(zscore, ddof=1)
data['z_mean_fr'] = data.groupby(['species','brain_region']).mean_fr.transform(zscore, ddof=1)
data['z_mean_norm_fr'] = data.groupby(['species','brain_region']).mean_norm_fr.transform(zscore, ddof=1)
data['z_fano'] = data.groupby(['species','brain_region']).fano.transform(zscore, ddof=1)
data['z_prop_burst'] = data.groupby(['species','brain_region']).prop_burst.transform(zscore, ddof=1)
data['z_prop_pause'] = data.groupby(['species','brain_region']).prop_pause.transform(zscore, ddof=1)
#%% get coefficients and p-values for each predictor

cis2 = []

model = smf.ols('tau ~ species + brain_region + z_n_spikes',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis2.append(('n_spikes',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('tau ~ species + brain_region + z_mean_fr',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis2.append(('mean_fr',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('tau ~ species + brain_region + z_mean_norm_fr',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis2.append(('mean_norm_fr',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('tau ~ species + brain_region + z_fano',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis2.append(('fano',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('tau ~ species + brain_region + z_prop_burst',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis2.append(('prop_burst',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('tau ~ species + brain_region + z_prop_pause',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis2.append(('prop_pause',ci.iloc[-1][0],ci.iloc[-1][1],beta))

cis2 = pd.DataFrame(cis2,columns=['parameter','ci_start','ci_end','beta'])

#%%

plt.figure(figsize=(4.5,2))

plt.errorbar(data=cis2,x='parameter',y='beta',yerr=(cis2['beta']-cis2['ci_start']),fmt='o')

plt.axhline(0.0,linestyle='dashed',color='red',linewidth=0.5)

plt.title('Timescale',size=7)

plt.ylabel('regression weight (a.u.)',fontsize=7)
plt.yticks(fontsize=7)

plt.xticks(range(6),['n spikes','mean FR','norm FR','fano','prop burst','prop pause'],fontsize=7)

plt.show()
#%% get coefficients and p-values for each predictor

cis = []

model = smf.ols('lat ~ species + brain_region + z_n_spikes',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis.append(('n_spikes',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('lat ~ species + brain_region + z_mean_fr',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis.append(('mean_fr',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('lat ~ species + brain_region + z_mean_norm_fr',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis.append(('mean_norm_fr',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('lat ~ species + brain_region + z_fano',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis.append(('fano',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('lat ~ species + brain_region + z_prop_burst',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis.append(('prop_burst',ci.iloc[-1][0],ci.iloc[-1][1],beta))

#

model = smf.ols('lat ~ species + brain_region + z_prop_pause',data=data)
res = model.fit()

ci = res.conf_int(alpha=0.05, cols=None)
beta = res.params[-1]

cis.append(('prop_pause',ci.iloc[-1][0],ci.iloc[-1][1],beta))

cis = pd.DataFrame(cis,columns=['parameter','ci_start','ci_end','beta'])

#%%

plt.figure(figsize=(4.5,2))

plt.errorbar(data=cis,x='parameter',y='beta',yerr=(cis['beta']-cis['ci_start']),fmt='o')

plt.axhline(0.0,linestyle='dashed',color='red',linewidth=0.5)

plt.title('Latency',size=7)

plt.ylabel('regression weight (a.u.)',fontsize=7)
plt.ylim(-6,6)
plt.yticks(fontsize=7)

plt.xticks(range(6),['n spikes','mean FR','norm FR','fano','prop burst','prop pause'],fontsize=7)

plt.show()
# %%
