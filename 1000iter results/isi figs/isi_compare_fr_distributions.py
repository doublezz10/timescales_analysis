#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'

#%% Load single-unit data

listofspecies = ['mouse','monkey','human']

fred_data = pd.read_csv('/Users/zachz/Library/CloudStorage/Box-Box/Timescales across species/filtered_isi_data.csv')

fred_data['species'] = pd.Categorical(fred_data['species'], categories=listofspecies, ordered=True)

zscore = lambda x: (x - x.mean()) / x.std()

fred_brain_region_data = fred_data

fred_brain_region_data.insert(14, 'zscore_fr', fred_brain_region_data.groupby(['species'])['FR'].transform(zscore))
fred_brain_region_data.insert(14, 'zscore_fr_ds', fred_brain_region_data.groupby(['dataset'])['FR'].transform(zscore))

#%%

sns.violinplot(data=fred_brain_region_data,x='species',y='FR')

model = smf.ols('FR ~ species',data=fred_brain_region_data)

res = model.fit()

print(res.summary())

#%%

sns.violinplot(data=fred_brain_region_data,x='species',y='zscore_fr')

model = smf.ols('FR ~ species',data=fred_brain_region_data)

res = model.fit()

print(res.summary())

#%%

sns.violinplot(data=fred_brain_region_data,x='species',y='zscore_fr_ds')

model = smf.ols('FR ~ species',data=fred_brain_region_data)

res = model.fit()

print(res.summary())
# %%
