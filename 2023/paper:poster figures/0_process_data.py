#%%

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

data = pd.read_csv('/Users/zachz/Library/Mobile Documents/com~apple~CloudDocs/Zach/Final Timescales Data/final_data.csv')

#%%

amyg = data[(data.area=='amygdala') | (data.area=='bla') | (data.area=='amyg') | (data.area=='AMG')]
hc = data[(data.area=='hippocampus') | (data.area=='hippocampus2') | (data.area=='dg') | (data.area=='ca1') | (data.area=='ca2') | (data.area=='ca3') | (data.area=='hc')]
acc = data[(data.area=='mcc') | (data.area=='dACC') | (data.area=='aca') | (data.area=='ACC')]
mpfc = data[(data.area=='scACC') | (data.area=='ila') | (data.area=='pl')]
ofc = data[(data.area=='OFC') | (data.area=='orb') | (data.area=='a11l') | (data.area=='a11m') | (data.area=='a13l') | (data.area=='a13m')]

acc_g = acc.assign(brain_region = 'ACC')
amyg_g = amyg.assign(brain_region = 'Amygdala')
hc_g = hc.assign(brain_region = 'Hippocampus')
mpfc_g = mpfc.assign(brain_region = 'mPFC')
ofc_g = ofc.assign(brain_region = 'OFC')

grouped_data = pd.concat((acc_g,amyg_g,hc_g,mpfc_g,ofc_g))

brain_regions = ['Hippocampus','Amygdala','OFC','mPFC','ACC']

grouped_data['brain_region'] = pd.Categorical(grouped_data['brain_region'], categories = brain_regions , ordered = True)

#%%

grouped_data.to_csv('processed_data.csv',index=False)
# %%
