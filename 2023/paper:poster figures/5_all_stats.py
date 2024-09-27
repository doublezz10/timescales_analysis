#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

from scipy import stats

plt.style.use('seaborn')

plt.rcParams['font.size'] = '7'

#%% Load single-unit data

data = pd.read_csv('processed_data.csv')

data['species'] = pd.Categorical(data['species'], categories = ['monkey','mouse','human'] , ordered = True)

filt_data = data[np.logical_and(data.tau < 1000, data.r2 > 0.5)]
filt_data = filt_data[filt_data.tau > 10]

brain_regions = ['Hippocampus','OFC','Amygdala','mPFC','ACC']

filt_data['brain_region'] = pd.Categorical(filt_data['brain_region'], categories = brain_regions , ordered = True)

fred_brain_region_data = filt_data

fred_data_lai = data[data.name=='stoll']

fred_data_lai['species'] = pd.Categorical(fred_data_lai['species'], categories = ['mouse','monkey','human'] , ordered = True)

#%%

model = smf.ols('tau~lat',data=fred_brain_region_data)

res = model.fit()

anova = sm.stats.anova_lm(res, typ=2)

anova

#%% Single-unit hierarchy

model = smf.ols('tau ~ species * brain_region',data=fred_brain_region_data)

res = model.fit()

anova = sm.stats.anova_lm(res, typ=1)

anova

#%%

model2 = smf.ols('tau ~ species + brain_region + FR',data=fred_brain_region_data)

res2 = model2.fit()

anova = sm.stats.anova_lm(res2, typ=2)

anova

#%%

print(anova_lm(res,res2))

#%% Single-unit hierarchy in lat

model = smf.ols('lat ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

anova = sm.stats.anova_lm(res, typ=2)

anova

#%%

model2 = smf.ols('lat ~ species + brain_region + FR',data=fred_brain_region_data)

res2 = model2.fit()

anova = sm.stats.anova_lm(res2, typ=2)

anova

#%%

print(anova_lm(res,res2))

# %% Amyg only tau

model = smf.ols('tau ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='Amygdala'])

res = model.fit()

print(res.summary())

# %% Amyg only lat

model = smf.ols('lat ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='Amygdala'])

res = model.fit()

print(res.summary())

# %% OFC only tau

model = smf.ols('tau ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='OFC'])

res = model.fit()

print(res.summary())

# %% OFC only lat

model = smf.ols('lat ~ species',data=fred_brain_region_data[fred_brain_region_data.brain_region=='OFC'])

res = model.fit()

print(res.summary())


#%% Within OFC


data = pd.read_csv('final_data.csv')

data['species'] = pd.Categorical(data['species'], categories = ['mouse','monkey','human'] , ordered = True)

filt_data = data[np.logical_and(data.tau < 1000, data.r2 > 0.5)]
filt_data = filt_data[filt_data.tau > 10]

fred_data = filt_data[filt_data.name=='stoll']

ofc_lai_vl = pd.DataFrame()

for area_ in ['LAI','a11l','a11m','a12l','a12m','a12o','a12r','a13l','a13m']:
    
    ofc_lai_vl = pd.concat((ofc_lai_vl,fred_data[fred_data.area==area_]))

ofc_lai_vl['area'] = ofc_lai_vl['area'].str.replace('LAI','AI')

ofc_lai_vl.loc[ofc_lai_vl['area'] == 'a11m', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['area']== 'a11l', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['area']== 'a12m', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['area']== 'a12l', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['area']== 'a12o', 'granularity'] = 'granular'
ofc_lai_vl.loc[ofc_lai_vl['area']== 'a12r', 'granularity'] = 'dysgranular'
ofc_lai_vl.loc[ofc_lai_vl['area'] =='a13m', 'granularity'] = 'dysgranular'
ofc_lai_vl.loc[ofc_lai_vl['area'] == 'a13l', 'granularity'] = 'dysgranular'
ofc_lai_vl.loc[ofc_lai_vl['area'] =='AI', 'granularity'] = 'agranular'

ofc_lai_vl['granularity'] = pd.Categorical(ofc_lai_vl['granularity'], categories=['granular','dysgranular','agranular'], ordered=True)

#%%

model = smf.ols('tau ~ granularity', data = ofc_lai_vl)

res = model.fit()

anova = sm.stats.anova_lm(res, typ=2)

anova


#%%

model = smf.ols('tau ~ area', data = ofc_lai_vl)

res2 = model.fit()

anova = sm.stats.anova_lm(res2, typ=2)

anova

#%%

print(anova_lm(res,res2))

#%%

model = smf.ols('lat ~ granularity', data = ofc_lai_vl)

res = model.fit()

anova = sm.stats.anova_lm(res, typ=2)

anova


#%%

model = smf.ols('lat ~ area', data = ofc_lai_vl)

res2 = model.fit()

anova = sm.stats.anova_lm(res2, typ=2)

anova
#%%

print(anova_lm(res,res2))
# %% what if you z-score fr

zscore = lambda x: (x - x.mean()) / x.std()

fred_brain_region_data.insert(14, 'zscore_fr', fred_brain_region_data.groupby(['species'])['FR'].transform(zscore))


model = smf.ols('tau ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

print(res.summary())

#%%

model2 = smf.ols('tau ~ species + brain_region + zscore_fr',data=fred_brain_region_data)

res2 = model2.fit()

print(res2.summary())

#%%

print(anova_lm(res,res2))

#%% Single-unit hierarchy in lat

model = smf.ols('lat ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

anova = sm.stats.anova_lm(res, typ=2)

anova

#%%

model2 = smf.ols('lat ~ species + brain_region + zscore_fr',data=fred_brain_region_data)

res2 = model2.fit()

anova = sm.stats.anova_lm(res2, typ=2)

anova

#%%

print(anova_lm(res,res2))

#%%

fred_brain_region_data.insert(14, 'zscore_fr_ds', fred_brain_region_data.groupby(['name'])['FR'].transform(zscore))

model = smf.ols('tau ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

anova = sm.stats.anova_lm(res, typ=2)

anova

#%%

model2 = smf.ols('tau ~ species + brain_region + zscore_fr_ds',data=fred_brain_region_data)

res2 = model2.fit()

anova = sm.stats.anova_lm(res2, typ=2)

anova

#%%

print(anova_lm(res,res2))

#%%

model = smf.ols('lat ~ species + brain_region',data=fred_brain_region_data)

res = model.fit()

anova = sm.stats.anova_lm(res, typ=2)

anova

#%%

model2 = smf.ols('lat ~ species + brain_region + zscore_fr_ds',data=fred_brain_region_data)

res2 = model2.fit()

anova = sm.stats.anova_lm(res2, typ=2)

anova

#%%

print(anova_lm(res,res2))
# %%
